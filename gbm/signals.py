import numpy as np
import pandas as pd

from gbm.regime import _ticker_key

DEFAULT_SMA_DAYS = 200
MOMENTUM_TRADE_DAYS = 25
MOMENTUM_PICKS = 3
_NEVER_LIQUIDATE = {"none", "non"}


class AssetSignal:
    """How one asset decides whether its sleeve is invested or parked in cash."""

    def __init__(self, source, enter, exit, follows_index, never_liquidate, asset_type=None, exclusion=0, bucket=None):
        self.source = source
        self.enter = enter
        self.exit = exit
        self.follows_index = follows_index
        self.never_liquidate = never_liquidate
        self.asset_type = asset_type
        self.exclusion = int(exclusion)
        self.bucket = bucket


def resolve_asset_signals(labels, sma_exit, sma_enter, tracked_index=None, asset_type=None, sma_exclusion=None, bucket=None):
    """Pick each asset's rule, in descending priority.

    A tracked index ticker follows that index. The words none and non never
    liquidate. Otherwise the row's SMA enter and exit are used, and a missing
    pair falls back to a 200-day enter and a 200-day exit. SMA-exclusion is how
    many of the newest closes stay out of that average. A missing exclusion
    keeps every close in the window. A Bucket name groups momentum assets into
    one candidate. A blank bucket leaves that asset on its own.
    """
    labels = list(labels)
    exits = list(sma_exit)
    enters = list(sma_enter)
    tracked = [None] * len(labels) if tracked_index is None else list(tracked_index)
    kinds = [None] * len(labels) if asset_type is None else [_asset_kind(value) for value in asset_type]
    exclusions = [0] * len(labels) if sma_exclusion is None else list(sma_exclusion)
    buckets = [None] * len(labels) if bucket is None else [_bucket_name(value) for value in bucket]

    def windows(index):
        exit_days = _positive_days(exits[index])
        enter_days = _positive_days(enters[index])
        exclusion = _exclusion_days(exclusions[index])
        if exit_days is None or enter_days is None:
            return DEFAULT_SMA_DAYS, DEFAULT_SMA_DAYS, exclusion
        return exit_days, enter_days, exclusion

    signals = []
    for index, _label in enumerate(labels):
        flag = _tracked_flag(tracked[index])
        exit_days, enter_days, exclusion = windows(index)
        if flag == "none":
            signals.append(AssetSignal(index, enter_days, exit_days, False, True, kinds[index], exclusion, buckets[index]))
            continue
        if flag is not None:
            source = _index_of(labels, flag)
            if source != index:
                source_exit, source_enter, source_exclusion = windows(source)
                signals.append(AssetSignal(
                    source, source_enter, source_exit, True, False, kinds[index], source_exclusion, buckets[index]
                ))
                continue
        signals.append(AssetSignal(index, enter_days, exit_days, False, False, kinds[index], exclusion, buckets[index]))
    return signals


def root_index(signals, index):
    seen = set()
    while signals[index].follows_index and index not in seen:
        seen.add(index)
        index = signals[index].source
    return index


def cash_asset_index(labels, ticker="SGOV"):
    for index, label in enumerate(labels):
        if _ticker_key(label) == ticker:
            return index
    return None


def tactical_weights(targets, signals, distances, cash_index=None):
    """None assets keep their targets. The leftover weight goes to the group furthest above its enter average.

    Followers share their tracked index, and that group's own targets split the leftover weight.
    A group with no target weight cannot take it. When nothing is above its enter average, the
    leftover weight sits in the cash asset.
    """
    targets = np.asarray(targets, dtype=float).reshape(-1)
    distances = np.asarray(distances, dtype=float)
    if distances.ndim == 1:
        distances = distances.reshape(1, -1)
    num_paths, num_assets = distances.shape
    never = np.array([signal.never_liquidate for signal in signals], dtype=bool)
    weights = np.zeros((num_paths, num_assets))
    weights[:, never] = targets[never]
    budget = 1.0 - weights.sum(axis=1, keepdims=True)

    roots = np.array([root_index(signals, index) for index in range(num_assets)])
    group_roots = []
    for index in range(num_assets):
        if never[index]:
            continue
        root = int(roots[index])
        if root not in group_roots:
            group_roots.append(root)
    if not group_roots:
        return _place_unallocated(weights, budget, targets, never, cash_index)

    score = np.full((num_paths, len(group_roots)), -np.inf)
    share = np.zeros((len(group_roots), num_assets))
    for column, root in enumerate(group_roots):
        members = np.where((roots == root) & ~never)[0]
        total = targets[members].sum()
        if total <= 0:
            continue
        share[column, members] = targets[members] / total
        root_distance = distances[:, root]
        usable = np.isfinite(root_distance) & (root_distance > 0)
        score[usable, column] = root_distance[usable]

    best = np.argmax(score, axis=1)
    best_score = score[np.arange(num_paths), best]
    chosen = np.isfinite(best_score)
    weights[chosen] += budget[chosen] * share[best[chosen]]
    if np.any(~chosen):
        leftover = np.where(chosen[:, None], 0.0, budget)
        weights = _place_unallocated(weights, leftover, targets, never, cash_index)
    return weights


def _place_unallocated(weights, budget, targets, never, cash_index):
    budget = np.asarray(budget, dtype=float)
    if budget.ndim == 1:
        budget = budget[:, None]
    if cash_index is not None:
        weights[:, cash_index] += budget[:, 0]
        return weights
    tactical_targets = np.where(never, 0.0, np.maximum(targets, 0.0))
    total = tactical_targets.sum()
    if total > 0:
        weights += budget * (tactical_targets / total)
    return weights


class SignalBook:
    """SMA enter/exit state. Trading asks which sleeves are in, then reports market returns."""

    def __init__(self, signals, num_paths, cash_drift, cash_volatility, cash_index=None, trade_interval=MOMENTUM_TRADE_DAYS, top_num=MOMENTUM_PICKS, beat_cash=True):
        self.signals = list(signals)
        self.cash_drift = float(cash_drift)
        self.cash_volatility = float(cash_volatility)
        self.beat_cash = bool(beat_cash)
        self.cash_index = cash_index
        self.trade_interval = int(trade_interval)
        self.top_num = _as_pick_count(top_num)
        num_assets = len(self.signals)
        self.invested = np.ones((num_paths, num_assets), dtype=bool)
        self.momentum_mask = np.array([signal.asset_type == "momentum" for signal in self.signals])
        self.gated_mask = np.array([signal.asset_type == "gated" for signal in self.signals])
        self.alternate_mask = np.array([
            signal.asset_type in ("alternate", "gated") for signal in self.signals
        ])
        self.uses_momentum_sleeve = bool(self.momentum_mask.any())
        # No momentum sleeve, and every asset is an alternate or gated. The book
        # holds the spreadsheet targets. It does not rotate into a winner, and a
        # name with no target weight cannot force a band trade.
        self.static_allocation = len(self.signals) > 0 and bool(np.all(self.alternate_mask))
        self._candidates = []
        self._choice = np.full(num_paths, -1, dtype=int)
        self._slots = np.full((num_paths, self.top_num), -1, dtype=int)
        self._build_candidates()
        self.selection = np.full(num_paths, 0 if cash_index is None else cash_index, dtype=int)
        self.in_cash = np.zeros(num_paths, dtype=bool)
        self.was_in_cash = self.in_cash.copy()
        self.steps = 0
        self._ready = False
        self._columns = np.arange(num_assets)
        self._enter_lengths = np.array([signal.enter for signal in self.signals], dtype=int)
        self._exit_lengths = np.array([signal.exit for signal in self.signals], dtype=int)
        self._exclusion = np.array([signal.exclusion for signal in self.signals], dtype=int)
        self._enter_spans = self._enter_lengths + self._exclusion
        self._exit_spans = self._exit_lengths + self._exclusion
        self._max_window = int(max(self._enter_spans.max(), self._exit_spans.max()))
        self._history = np.ones((self._max_window, num_paths, num_assets))
        self._count = 1
        self._pos = 0
        self._steps_seen = 1
        self._follows = np.array([signal.follows_index for signal in self.signals], dtype=bool)
        self._never = np.array([signal.never_liquidate for signal in self.signals], dtype=bool)
        self._roots = np.array([root_index(self.signals, index) for index in range(num_assets)], dtype=int)
        self._momentum_asset = self.momentum_mask.copy()
        if cash_index is not None:
            self._momentum_asset[cash_index] = False
        self._enter_sum = self._history[0].copy()
        self._exit_sum = self._history[0].copy()
        self._recent_sum = np.zeros((num_paths, num_assets))
        excluded = self._exclusion > 0
        if np.any(excluded):
            self._recent_sum[:, excluded] = self._history[0][:, excluded]

    @property
    def lead_days(self):
        """Trading days of history before a path can leave cash.

        This is the longest SMA Enter or SMA Exit window. Closes that
        SMA-exclusion leaves out of the average are included, because that
        average does not exist until those closes do.
        """
        return int(self._max_window)

    def start(self, in_cash=True, asset=None):
        """Open the momentum sleeve in SGOV, or in one named asset."""
        if in_cash:
            if self.cash_index is None:
                raise ValueError("momentum sleeve needs SGOV")
            self.selection[:] = self.cash_index
            self._choice[:] = -1
            self._slots[:] = -1
            self.in_cash[:] = True
        else:
            asset = int(asset)
            self.selection[:] = asset
            self._choice[:] = self._candidate_column(asset)
            self._slots[:] = -1
            self.in_cash[:] = False
        self._ready = True

    def consider_trade(self, targets=None):
        """Momentum selection runs every `trade_interval` trading days."""
        self.steps += 1
        if self.steps % self.trade_interval != 0:
            return False
        self.was_in_cash = self.in_cash.copy()
        self.choose_momentum(targets)
        return True

    def choose_momentum(self, targets=None):
        """Hold the top `top_num` momentum candidates, each with an equal share of the sleeve.

        A candidate is one solo momentum asset, or a bucket whose gap is the allocation-weighted
        average of its members.         Invested paths measure the gap over SMA-Exit. Paths in SGOV
        measure it over SMA-Enter. Both averages leave out the newest SMA-exclusion closes.
        With beat_cash on, a candidate has to beat the SGOV rate to take a share. With it off,
        any candidate above its SMA qualifies. Empty shares stay in SGOV. When cash wins,
        nothing cleared that hurdle, so the whole sleeve is in SGOV and gated assets park
        there with it. A path with no average yet stays put only when nothing is ready;
        a ready gap that loses to the hurdle is a cash win.
        """
        if not self._candidates:
            return
        self._check_bucket_allocations(targets)
        exit_ranking = self._score("exit", targets)
        enter_ranking = self._score("enter", targets)
        ranking = np.where(self.in_cash[:, None], enter_ranking, exit_ranking)
        order = np.argsort(-ranking, axis=1, kind="stable")
        num_paths = ranking.shape[0]
        slots = np.full((num_paths, self.top_num), -1, dtype=int)
        picks = min(self.top_num, order.shape[1])
        hurdle = self.cash_drift if self.beat_cash else 0.0
        for slot in range(picks):
            choice = order[:, slot]
            score = ranking[np.arange(num_paths), choice]
            keep = np.isfinite(score) & (score > hurdle)
            slots[keep, slot] = choice[keep]
        ready = np.isfinite(ranking).any(axis=1)
        if not np.any(ready):
            return
        self._slots[ready] = slots[ready]
        self._choice[ready] = -1
        heads = np.array([members[0] for _key, members in self._candidates], dtype=int)
        first = np.full(num_paths, -1, dtype=int)
        for slot in range(self.top_num):
            choice = slots[:, slot]
            take = (first < 0) & (choice >= 0)
            first[take] = heads[choice[take]]
        held = ready & (first >= 0)
        give_up = ready & ~held
        needs_cash = give_up | (ready & (slots < 0).any(axis=1))
        if np.any(needs_cash):
            if self.cash_index is None:
                raise ValueError("momentum sleeve needs SGOV when a pick is left in cash")
        self.in_cash[held] = False
        self.in_cash[give_up] = True
        if np.any(held):
            self.selection[held] = first[held]
        if np.any(give_up):
            self.selection[give_up] = self.cash_index

    def _score(self, kind, targets=None):
        distance = self._momentum_distance(kind)
        num_paths = distance.shape[0]
        ranking = np.full((num_paths, len(self._candidates)), -np.inf)
        target_row = None if targets is None else np.asarray(targets, dtype=float).reshape(-1)
        for column, (_key, members) in enumerate(self._candidates):
            member_distance = distance[:, members]
            ready = np.isfinite(member_distance).all(axis=1)
            if target_row is None:
                weights = np.ones(len(members))
            else:
                weights = target_row[list(members)]
            total = float(weights.sum())
            if total <= 0:
                weighted = member_distance.mean(axis=1)
            else:
                weighted = member_distance @ weights / total
            ranking[ready, column] = weighted[ready]
        return ranking

    def _momentum_distance(self, kind):
        """Percent each risky momentum asset sits above its own enter or exit average."""
        num_paths, num_assets = self.invested.shape
        distance = np.full((num_paths, num_assets), np.nan)
        if kind == "exit":
            lengths, sums, spans = self._exit_lengths, self._exit_sum, self._exit_spans
        else:
            lengths, sums, spans = self._enter_lengths, self._enter_sum, self._enter_spans
        average = self._window_mean(sums, lengths)
        price = self._history[self._pos]
        usable = self._momentum_asset & (self._count >= spans) & (average > 0)
        distance[usable] = (price[usable] - average[usable]) / average[usable]
        return distance

    def held(self):
        """None assets stay on. The tactical group furthest above its enter average is on with them.

        An all-alternate book has no tactical group. Every asset stays invested
        and earns its own drift.
        """
        if self.static_allocation:
            return np.ones(self.invested.shape, dtype=bool)
        distances = self.enter_distances()
        num_paths, num_assets = distances.shape
        never = np.array([signal.never_liquidate for signal in self.signals])
        held = np.broadcast_to(never, (num_paths, num_assets)).copy()
        roots = np.array([root_index(self.signals, index) for index in range(num_assets)])
        group_roots = []
        for index in range(num_assets):
            if never[index]:
                continue
            root = int(roots[index])
            if root not in group_roots:
                group_roots.append(root)
        if not group_roots:
            return held
        score = np.column_stack([distances[:, root] for root in group_roots])
        score = np.where(np.isfinite(score) & (score > 0), score, -np.inf)
        best = np.argmax(score, axis=1)
        best_score = score[np.arange(num_paths), best]
        winning = np.isfinite(best_score)
        for column, root in enumerate(group_roots):
            members = (roots == root) & ~never
            paths = winning & (best == column)
            held[paths[:, None] & members[None, :]] = True
        return held

    def enter_distances(self):
        num_paths, num_assets = self.invested.shape
        distances = np.full((num_paths, num_assets), np.nan)
        average = self._window_mean(self._enter_sum, self._enter_lengths)
        price = self._history[self._pos]
        usable = ~self._follows & ~self._never & (self._count >= self._enter_spans) & (average > 0)
        distances[usable] = (price[usable] - average[usable]) / average[usable]
        if np.any(self._follows):
            distances[:, self._follows] = distances[:, self._roots[self._follows]]
        return distances

    def target_weights(self, targets):
        """Momentum sleeve plus alternates. Gated alternates fold into SGOV when cash wins.

        After a ranking, the top `top_num` candidates each take an equal share of the momentum sleeve.
        A bucket splits its share by member target weights. Shares with no qualifying name stay
        in SGOV. When cash wins, the whole sleeve is in SGOV and gated assets park there too.
        A book with no momentum sleeve keeps the tactical rotation, unless every asset is an
        alternate. That book holds the given targets on every path.
        """
        if not self.uses_momentum_sleeve:
            if self.static_allocation:
                row = np.asarray(targets, dtype=float).reshape(-1)
                return np.broadcast_to(row, (self.invested.shape[0], row.shape[0])).copy()
            return tactical_weights(targets, self.signals, self.enter_distances(), self.cash_index)
        targets = np.asarray(targets, dtype=float).reshape(-1)
        weights = np.zeros((self.selection.shape[0], targets.size))
        weights[:, self.alternate_mask] = targets[self.alternate_mask]
        sleeve = 1.0 - float(targets[self.alternate_mask].sum())
        weights += sleeve * self.sleeve_weights(targets)
        parked = self.in_cash
        if self.gated_mask.any() and np.any(parked):
            weights[parked[:, None] & self.gated_mask] = 0.0
            gated_budget = float(targets[self.gated_mask].sum())
            rows = np.flatnonzero(parked)
            weights[rows, self.selection[rows]] = weights[rows, self.selection[rows]] + gated_budget
        return weights

    def apply(self, drift, volatility):
        drift = np.array(drift, dtype=float, copy=True)
        volatility = np.array(volatility, dtype=float, copy=True)
        out = ~self.held()
        drift[out] = self.cash_drift
        volatility[out] = self.cash_volatility
        return drift, volatility

    def load_history(self, closes):
        """Replace the ring with `closes`, oldest first. The last row is the current price."""
        closes = np.asarray(closes, dtype=float)
        if closes.ndim == 2:
            closes = closes[:, None, :]
        if closes.ndim != 3:
            raise ValueError("closes must be days by assets, or days by paths by assets")
        capacity, num_paths, num_assets = self._history.shape
        if closes.shape[2] != num_assets:
            raise ValueError("closes have the wrong number of assets")
        if closes.shape[0] > capacity:
            closes = closes[-capacity:]
        if closes.shape[1] == 1 and num_paths != 1:
            closes = np.broadcast_to(closes, (closes.shape[0], num_paths, num_assets))
        elif closes.shape[1] != num_paths:
            raise ValueError("closes have the wrong number of paths")
        count = int(closes.shape[0])
        if count < 1:
            raise ValueError("closes must include a price")
        self._history = np.ones_like(self._history)
        self._history[:count] = closes
        self._count = count
        self._pos = count - 1
        self._sync_window_sums()

    def observe(self, market_growth):
        growth = np.asarray(market_growth, dtype=float)
        new_pos = (self._pos + 1) % self._max_window
        enter_full = self._steps_seen >= self._enter_spans
        exit_full = self._steps_seen >= self._exit_spans
        excluded = self._exclusion > 0
        recent_full = excluded & (self._steps_seen >= self._exclusion)
        enter_dropped = self._prices_at(new_pos, self._enter_spans)
        exit_dropped = self._prices_at(new_pos, self._exit_spans)
        recent_dropped = self._prices_at(new_pos, self._exclusion) if np.any(recent_full) else None
        previous = self._history[(new_pos - 1) % self._max_window]
        new_price = previous * growth
        self._history[new_pos] = new_price
        self._pos = new_pos
        self._steps_seen += 1
        self._count = min(self._steps_seen, self._max_window)
        self._enter_sum += new_price
        self._exit_sum += new_price
        if np.any(enter_full):
            self._enter_sum[:, enter_full] -= enter_dropped[:, enter_full]
        if np.any(exit_full):
            self._exit_sum[:, exit_full] -= exit_dropped[:, exit_full]
        if np.any(excluded):
            self._recent_sum[:, excluded] += new_price[:, excluded]
            if recent_dropped is not None:
                self._recent_sum[:, recent_full] -= recent_dropped[:, recent_full]
        self._update()

    def _prices_at(self, new_pos, lengths):
        slots = (new_pos - lengths) % self._max_window
        return np.swapaxes(self._history[slots, :, self._columns], 0, 1)

    def _window_mean(self, sums, lengths):
        """Mean of `lengths` closes, after the newest exclusion closes are removed."""
        body = sums - self._recent_sum
        return np.divide(body, lengths, out=np.zeros_like(sums), where=lengths > 0)

    def _sync_window_sums(self):
        """Rebuild running sums from the prices currently stored in the ring."""
        self._steps_seen = self._count
        self._enter_sum = self._gathered_sum(self._enter_spans)
        self._exit_sum = self._gathered_sum(self._exit_spans)
        self._recent_sum = self._gathered_sum(self._exclusion)

    def _gathered_sum(self, lengths):
        num_paths, num_assets = self.invested.shape
        total = np.zeros((num_paths, num_assets))
        for length in np.unique(lengths):
            length = int(length)
            columns = lengths == length
            span = length if self._count >= length else self._count
            if span <= 0:
                continue
            total[:, columns] = self._window(span).sum(axis=0)[:, columns]
        return total

    def _root(self, index):
        return root_index(self.signals, index)

    def _window(self, length):
        offsets = np.arange(length - 1, -1, -1)
        return self._history[(self._pos - offsets) % self._max_window]

    def _update(self):
        price = self._history[self._pos]
        was_in = self.invested
        tracked = ~self._follows & ~self._never
        exit_average = self._window_mean(self._exit_sum, self._exit_lengths)
        enter_average = self._window_mean(self._enter_sum, self._enter_lengths)
        exit_ready = tracked & (self._count >= self._exit_spans)
        enter_ready = tracked & (self._count >= self._enter_spans)
        exited = exit_ready & was_in & (price < exit_average)
        entered = enter_ready & ~was_in & (price > enter_average)
        updated = np.where(exited, False, was_in)
        self.invested = np.where(entered, True, updated)

    @property
    def has_buckets(self):
        return any(key[0] == "bucket" for key, _members in self._candidates)

    def selected_name(self, labels):
        """Top bucket or ticker on path 0. Cash when cash won."""
        names = self.selected_names(labels)
        return names[0] if names else labels[int(self.selection[0])]

    def selected_names(self, labels):
        """Bucket names and solo tickers held on path 0, best first."""
        labels = list(labels)
        if bool(self.in_cash[0]):
            return [labels[int(self.selection[0])]]
        names = []
        for choice in self._slots[0]:
            if int(choice) < 0:
                continue
            names.append(self._candidate_label(labels, int(choice)))
        if names:
            return names
        choice = int(self._choice[0])
        if choice >= 0:
            return [self._candidate_label(labels, choice)]
        return [labels[int(self.selection[0])]]

    def holding_mask(self):
        """Assets currently carrying the momentum sleeve. A bucket marks every member."""
        mask = np.zeros((self.selection.shape[0], len(self.signals)), dtype=bool)
        ranked = (self._slots >= 0).any(axis=1)
        self._mark_slots(mask, ranked)
        if self.cash_index is not None:
            residual = ranked & (self._slots < 0).any(axis=1)
            mask[residual, self.cash_index] = True
        unresolved = np.flatnonzero(~ranked)
        if unresolved.size:
            mask[unresolved, self.selection[unresolved]] = True
        return mask

    def sleeve_weights(self, targets):
        """Fraction of the momentum sleeve each asset holds. Each row sums to 1.

        A ranked book gives an equal share to each of the top `top_num` names. A bucket splits its
        share by member targets. Before a ranking, the single open holding takes the whole sleeve.
        """
        self._check_bucket_allocations(targets)
        target_row = None if targets is None else np.asarray(targets, dtype=float).reshape(-1)
        weights = np.zeros((self.selection.shape[0], len(self.signals)))
        ranked = (self._slots >= 0).any(axis=1)
        share_of_sleeve = 1.0 / self.top_num
        for column, (_key, members) in enumerate(self._candidates):
            share = self._member_share(members, target_row)
            members = list(members)
            for slot in range(self.top_num):
                rows = np.flatnonzero(ranked & (self._slots[:, slot] == column))
                if rows.size == 0:
                    continue
                weights[np.ix_(rows, members)] += share_of_sleeve * share
        if self.cash_index is not None and np.any(ranked):
            empty = (self._slots < 0).sum(axis=1)
            weights[ranked, self.cash_index] += empty[ranked] * share_of_sleeve
        unranked = ~ranked
        unresolved = np.flatnonzero(unranked & (self._choice < 0))
        if unresolved.size:
            weights[unresolved, self.selection[unresolved]] = 1.0
        for column, (_key, members) in enumerate(self._candidates):
            rows = np.flatnonzero(unranked & (self._choice == column))
            if rows.size == 0:
                continue
            share = self._member_share(members, target_row)
            weights[np.ix_(rows, list(members))] = share
        return weights

    def _mark_slots(self, mask, ranked):
        for column, (_key, members) in enumerate(self._candidates):
            members = list(members)
            for slot in range(self.top_num):
                rows = np.flatnonzero(ranked & (self._slots[:, slot] == column))
                if rows.size == 0:
                    continue
                mask[np.ix_(rows, members)] = True

    def _candidate_label(self, labels, column):
        key, members = self._candidates[column]
        if key[0] == "bucket":
            return key[1]
        return labels[members[0]]

    def _member_share(self, members, target_row):
        members = list(members)
        if target_row is None or len(members) == 1:
            return np.full(len(members), 1.0 / len(members))
        raw = np.asarray(target_row, dtype=float)[members]
        total = float(raw.sum())
        if total <= 0:
            return np.full(len(members), 1.0 / len(members))
        return raw / total

    def _build_candidates(self):
        groups = {}
        order = []
        for index, signal in enumerate(self.signals):
            bucket = signal.bucket
            if bucket and index == self.cash_index:
                raise ValueError("SGOV cannot belong to a momentum bucket")
            if bucket and signal.asset_type != "momentum":
                raise ValueError(f"bucket {bucket} is set on an asset that is not momentum")
            if signal.asset_type != "momentum" or index == self.cash_index:
                continue
            if bucket:
                key = ("bucket", bucket)
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(index)
            else:
                key = ("solo", index)
                groups[key] = [index]
                order.append(key)
        self._candidates = [(key, groups[key]) for key in order]

    def _candidate_column(self, asset):
        for column, (_key, members) in enumerate(self._candidates):
            if asset in members:
                return column
        return -1

    def _check_bucket_allocations(self, targets):
        """Each bucket's targets have to add up to the momentum portion of the book."""
        if targets is None or not self.has_buckets:
            return
        targets = np.asarray(targets, dtype=float).reshape(-1)
        portion = 1.0 - float(targets[self.alternate_mask].sum())
        for key, members in self._candidates:
            if key[0] != "bucket":
                continue
            total = float(targets[list(members)].sum())
            if not np.isclose(total, portion, rtol=0.0, atol=1e-8):
                raise ValueError(
                    f"bucket {key[1]} allocations sum to {total:.4f}, momentum portion is {portion:.4f}"
                )


def current_momentum_bucket(
    labels,
    sma_exit,
    sma_enter,
    cash_drift,
    tracked_index=None,
    asset_type=None,
    sma_exclusion=None,
    prices=None,
    download=None,
    bucket=None,
    targets=None,
):
    """Recommend the momentum bucket from the newest SMA-Enter closes.

    The sleeve is scored from cash, so `choose_momentum` measures each gap over
    SMA-Enter. That average leaves out the newest SMA-exclusion closes, and the
    history keeps enough sessions to fill the longest of those windows. A named
    bucket is one candidate, scored as the allocation-weighted gap of its members.
    """
    labels = list(labels)
    signals = resolve_asset_signals(
        labels, sma_exit, sma_enter, tracked_index, asset_type, sma_exclusion, bucket
    )
    window = max(signal.enter + signal.exclusion for signal in signals)
    closes = _recent_closes(_price_frame(labels, prices, download), labels, window)
    book = SignalBook(
        signals,
        num_paths=1,
        cash_drift=cash_drift,
        cash_volatility=0.0,
        cash_index=cash_asset_index(labels),
    )
    _load_closes(book, closes)
    book.start(in_cash=True)
    book.choose_momentum(targets)
    return book.selected_name(labels)


def _price_frame(labels, prices, download):
    if prices is not None:
        return prices
    if download is None:
        from gbm.asset_simulation_matrix import download_prices

        download = download_prices
    symbols = []
    for label in labels:
        symbol = _price_symbol(label)
        if symbol not in symbols:
            symbols.append(symbol)
    return download(symbols)


def _price_symbol(label):
    """Market symbol for a sleeve. VOO/VIIX prices as VOO, and BTC keeps the spot backfill."""
    text = str(label).strip()
    if text in ("BTC", "IBIT"):
        return "BTC"
    parts = [part.strip() for part in text.split("/") if part.strip()]
    return parts[0] if parts else text


def _recent_closes(raw, labels, window):
    """Last `window` closes, one column per sleeve, oldest first."""
    if isinstance(raw, np.ndarray):
        frame = pd.DataFrame(np.asarray(raw, dtype=float), columns=list(labels))
    else:
        frame = raw
    columns = []
    for label in labels:
        symbol = _price_symbol(label)
        if label in getattr(frame, "columns", ()):
            series = frame[label]
        elif symbol in frame.columns:
            series = frame[symbol]
        else:
            raise KeyError(f"{label} has no closes")
        columns.append(pd.Series(series.to_numpy(dtype=float), name=label))
    prices = pd.concat(columns, axis=1).dropna()
    if prices.empty:
        raise ValueError("no closes for the momentum window")
    return prices.tail(window).to_numpy(dtype=float)


def _load_closes(book, closes):
    """Replace the ring with `closes` so the newest row is the current price."""
    book.load_history(closes)


def _as_pick_count(value):
    """How many momentum buckets or solo names the sleeve holds."""
    if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, (int, np.integer)):
        raise ValueError("top_num must be a positive whole number")
    count = int(value)
    if count < 1:
        raise ValueError("top_num must be a positive whole number")
    return count


def _bucket_name(value):
    """Bucket label. A blank cell leaves the asset on its own."""
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text


def _asset_kind(value):
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    text = str(value).strip().lower()
    if text in ("", "nan"):
        return None
    if text == "alternative":
        return "alternate"
    if text in ("momentum", "alternate", "gated"):
        return text
    raise ValueError(f"unknown asset type {value}")


def _exclusion_days(value):
    """How many newest closes stay out of the average. Missing or non-positive means none."""
    days = _positive_days(value)
    return 0 if days is None else days


def _positive_days(value):
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    try:
        days = int(float(value))
    except (TypeError, ValueError):
        return None
    if days <= 0:
        return None
    return days


def _tracked_flag(value):
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    if text.lower() in _NEVER_LIQUIDATE:
        return "none"
    return text


def _index_of(labels, name):
    key = _ticker_key(name)
    for index, label in enumerate(labels):
        if _ticker_key(label) == key:
            return index
    raise KeyError(f"{name} is not a portfolio asset")
