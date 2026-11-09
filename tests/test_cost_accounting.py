"""Price-day rule, cost formula, savings and break-even, request parsing."""
import datetime as dt
import json

import pytest

from llmdw.benchmark import eval_time
from llmdw.scoring import cluster, cost_accounting as g

P = {"input": 2.0, "cached_input": 0.1, "output": 10.0}


def _src(date, model="m", time="", kind="Internet Archive snapshot", price=None):
    return {"date": date, "time_utc": time, "kind": kind, "folder": f"f_{date}_{time}",
            "prices": {model: price or P}}


def test_price_day_rule():
    S = [_src("2026-10-30", time="2026-10-30T10:00"), _src("2026-10-30", time="2026-10-30T20:00"),
         _src("2026-11-02")]
    s, rule = g.choose_source("m", "2026-10-30", S)
    assert rule == "on the day" and s["time_utc"] == "2026-10-30T20:00"      # the last one of the day
    s, rule = g.choose_source("m", "2026-11-01", S)
    assert rule == "nearest source before the day" and s["date"] == "2026-10-30"
    s, rule = g.choose_source("m", "2026-10-29", S)
    assert rule == "nearest source after the day" and s["date"] == "2026-10-30" and s["time_utc"].endswith("10:00")
    s, rule = g.choose_source("m", "2026-11-05", S)
    assert rule == "nearest source before the day" and s["date"] == "2026-11-02"
    # no source with a price for the model: the capture of the fallback day
    with_other = S + [_src(g.FALLBACK_DAY, model="other", kind="own capture")]
    s, rule = g.choose_source("other_model", "2026-11-01", with_other)
    assert rule == g.RULES[3] and s["date"] == g.FALLBACK_DAY


def test_cost_formula_with_cached_and_cache_write_tokens():
    tok = {"prompt_tokens": 1_000_000, "cached_prompt_tokens": 0, "cache_write_5m_tokens": 0,
           "cache_write_1h_tokens": 0, "completion_tokens": 100_000}
    r = {**P, "cache_write_5m": 2.5, "cache_write_1h": 4.0}
    assert g.cost_usd(tok, r) == pytest.approx(2.0 + 1.0)
    tok["cached_prompt_tokens"] = 400_000                 # 0.6 M at 2, 0.4 M at 0.1, output 0.1 M at 10
    assert g.cost_usd(tok, r) == pytest.approx(1.2 + 0.04 + 1.0)
    tok.update(cached_prompt_tokens=0, cache_write_5m_tokens=200_000, cache_write_1h_tokens=100_000)
    assert g.cost_usd(tok, r) == pytest.approx(0.7 * 2.0 + 0.2 * 2.5 + 0.1 * 4.0 + 1.0)
    tok.update(cache_write_5m_tokens=2_000_000)
    with pytest.raises(ValueError):
        g.cost_usd(tok, r)


def test_cached_price_only_where_usage_records_cached_tokens():
    u = {"prompt_tokens": 1000, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}}
    t = g.usage_tokens(u)
    assert t["cached_prompt_tokens"] == 0
    assert g.cost_usd(t, {"input": 2.0, "cached_input": None, "output": 10.0}) == pytest.approx((1000 * 2 + 100) / 1e6)
    u["prompt_tokens_details"]["cached_tokens"] = 600
    t = g.usage_tokens(u)
    assert g.cost_usd(t, P) == pytest.approx((400 * 2 + 600 * 0.1 + 100) / 1e6)


def test_claude_cache_write_tokens_from_the_billing_usage():
    u = {"prompt_tokens": 4619, "completion_tokens": 5,
         "billing_usage": {"claude_usage": {"input_tokens": 4, "cache_creation_input_tokens": 4615,
                                            "cache_read_input_tokens": 0,
                                            "cache_creation": {"ephemeral_5m_input_tokens": 4615}}}}
    t = g.usage_tokens(u)
    assert (t["cache_write_5m_tokens"], t["cache_write_1h_tokens"], t["cached_prompt_tokens"]) == (4615, 0, 0)


def test_time_of_day_rates():
    p = {"input": 1.32, "cached_input": 0.044, "output": 3.96, "offpeak_input": 0.66, "offpeak_cached_input": 0.022,
         "offpeak_output": 1.98, "peak_hours_utc": [[1, 4], [6, 10]], "peak_weekdays": [0, 1, 2, 3, 4]}
    fri = dt.datetime(2026, 10, 30, 7, 30, tzinfo=dt.timezone.utc)
    assert g.rates(p, fri)[0] == "peak" and g.rates(p, fri)[1]["input"] == 1.32
    assert g.rates(p, fri.replace(hour=4))[0] == "off-peak"                    # 04:00 ends the first window
    assert g.rates(p, fri.replace(hour=12))[1]["output"] == 1.98
    sat = dt.datetime(2026, 10, 31, 7, 30, tzinfo=dt.timezone.utc)
    assert g.rates(p, sat)[0] == "off-peak"
    assert g.rates(P, fri)[0] == "standard"


def test_request_day_is_the_local_date_of_the_start():
    r = {"time": "2026-11-01T00:00:30", "latency": 60}
    t0 = g.start_time(r)
    assert t0.date().isoformat() == "2026-10-31"
    assert t0.astimezone(dt.timezone.utc).hour == 4                              # CDT, UTC - 5


def test_tags_and_presentations():
    t = g.parse_tag("with_data:rich12_lot_sizing_3:original:s2")
    assert (t["prompt"], t["instance"], t["condition"], t["sample"]) == ("with_data", "rich12_lot_sizing_3",
                                                                        "original", 2)
    assert g.presentation_of(t["prompt"], t["condition"]) == "data"
    assert g.presentation_of("support", "original") == "original"
    assert g.presentation_of("ranked", "original") == "ranked, original"
    assert g.error_category("incomplete: no reasoning (neither ...)") == "incomplete: no reasoning"


def test_sign_of_savings_and_break_even():
    assert g.saved(10.0, 4.0) == 6.0              # the baseline needs 10 evaluations, the order 4: 6 saved
    assert g.saved(4.0, 10.0) == -6.0
    assert g.saved(None, 3.0) is None
    assert g.break_even(30.0, 6.0) == 5.0         # 30 s of latency against 6 evaluations: 5 s per evaluation
    assert g.break_even(30.0, 0.0) is None and g.break_even(30.0, -1.0) is None


def test_net_row():
    cluster.set_groups({})
    saved = {"a": 4.0, "b": 6.0, "c": 5.0, "d": None}
    lat = {"a": 10.0, "b": 20.0, "c": 30.0, "d": 1.0}
    cost = {"a": 0.1, "b": 0.2, "c": 0.3}
    r = g.net_row(saved, lat, cost, t_eval=2.0)
    assert r["n_instances"] == 3 and r["n_groups"] == 3
    assert r["evaluations_saved_mean"] == pytest.approx(5.0)
    assert r["solver_seconds_saved_collection_time_mean"] == pytest.approx(10.0)
    assert r["latency_mean_s"] == pytest.approx(20.0)
    assert r["net_seconds_saved_collection_time_mean"] == pytest.approx(10.0 - 20.0)
    assert r["break_even_eval_seconds"] == pytest.approx(20.0 / 5.0)
    assert r["cost_mean_usd"] == pytest.approx(0.2)
    assert r["solver_seconds_saved_collection_time_se"] == pytest.approx(2.0 * r["evaluations_saved_se"])
    assert r["break_even_eval_seconds_se"] is not None
    r = g.net_row({"a": -1.0, "b": -2.0}, {"a": 1.0, "b": 1.0}, {}, t_eval=None)
    assert "break_even_eval_seconds" not in r and "net_seconds_saved_collection_time_mean" not in r
    # prompt groups: two instances of one group count as one group
    cluster.set_groups({"a": "g", "b": "g"})
    assert g.net_row(saved, lat, cost, 1.0)["n_groups"] == 2
    cluster.set_groups({})


def test_read_logs_drops_duplicate_lines_and_counts_requests(tmp_path):
    line = {"time": "2026-11-08T10:00:00", "tag": "support:x:original:s0", "request_sha256": "h",
            "model_requested": "gpt-6.1-sol", "model_returned": "gpt-6.1-sol", "latency": 3.0, "cached": False,
            "finish_reason": "stop", "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                                               "prompt_tokens_details": {"cached_tokens": 50}}}
    hit = {**line, "time": "2026-11-08T11:00:00", "cached": True}
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps(line) + "\n" + json.dumps(hit) + "\n")
    b.write_text(json.dumps(line) + "\n")
    lines, dup = g.read_logs([("x", a), ("y", b)])
    assert dup == 1 and len(lines) == 2
    prices = {("gpt-6.1-sol", "2026-11-08"): {**P, "rule": "on the day", "source": "s", "source_date": "2026-11-08",
                                              "url": "", "snapshot": ""}}
    req = g.request_rows(lines, prices, {"gpt-6.1-sol": "GPT-6.1 Sol"}, {"x": g.COLL_TEMPLATES})
    assert len(req) == 1 and req[0]["model"] == "GPT-6.1 Sol"
    assert req[0]["cost_usd"] == pytest.approx((50 * 2 + 50 * 0.1 + 10 * 10) / 1e6)
    cells = g.answer_cells(lines, req, prices, {"x": g.COLL_TEMPLATES})
    # the last line of the cell is the cache hit; it takes the latency and cost of the sent request
    (v,) = cells.values()
    assert v["from_cache"] and v["found"] and v["latency"] == 3.0 and v["cost"] == req[0]["cost_usd"]
    summ = g.summary_rows(req, lines)
    tot = next(r for r in summ if r["experiment"] == "all" and r["config"] == "all")
    assert tot["requests"] == 1 and tot["cache_hits"] == 1


def test_bnp_eval_time(tmp_path):
    p = tmp_path / "bnp_runs.csv"
    p.write_text("instance,class,set,seed,status,solved,time\n"
                 "i1,c,A,0,optimal,True,10\n"
                 "i1,c,A,1,timelimit,False,1805\n"
                 "i1,c,gcg_default,0,optimal,True,1\n"
                 "i2,c,A,0,optimal,True,30\n")
    rows = g.bnp_eval_time(p)
    allr = next(r for r in rows if r["class"] == "all")
    assert allr["n_sets"] == 3 and allr["n_at_limit"] == 1
    assert allr["mean_seconds"] == pytest.approx(((10 + 1800) / 2 + 30) / 2)


def test_eval_time_sets():
    assert "large" in eval_time.SETS and "rich_models_large" in eval_time.COLLECTIONS
    with pytest.raises(ValueError):
        eval_time.collect(None, [], 1, which="other")


def test_net_row_with_the_time_of_each_instance():
    from llmdw.scoring import cluster, cost_accounting as g

    cluster.set_groups({})
    saved = {"a": 2.0, "b": 1.0, "c": 0.0}
    lat = {"a": 10.0, "b": 10.0, "c": 10.0}
    r = g.net_row(saved, lat, {}, 4.0, {"a": 6.0, "b": 20.0, "c": 100.0})
    # with the collection's time: 8 - 10, 4 - 10, 0 - 10; with each instance's time: 12 - 10, 20 - 10, 0 - 10
    assert abs(r["net_seconds_saved_collection_time_mean"] - (-2 - 6 - 10) / 3) < 1e-12
    assert abs(r["net_seconds_saved_own_time_mean"] - (2 + 10 - 10) / 3) < 1e-12
    assert abs(r["share_positive_net_own_time"] - 2 / 3) < 1e-12 and r["n_instances_own_time"] == 3


def test_charge_of_the_five_dual_lookahead():
    from llmdw.scoring import cluster, cost_accounting as g

    cluster.set_groups({})
    saved, lat = {"a": 2.0, "b": 1.0}, {"a": 10.0, "b": 10.0}
    r = g.net_row(saved, lat, {}, 4.0, {"a": 6.0, "b": 20.0}, {"a": 3.0, "b": -1.0})
    assert abs(r["net_seconds_saved_own_time_mean"] - ((12 - 10 - 3) + (20 - 10 + 1)) / 2) < 1e-12
    assert abs(r["net_seconds_saved_collection_time_mean"] - ((8 - 10 - 3) + (4 - 10 + 1)) / 2) < 1e-12
    assert abs(r["break_even_eval_seconds"] - (10 + 1) / 1.5) < 1e-12
    assert abs(r["eval_seconds_own_mean"] - 13.0) < 1e-12 and abs(r["side_seconds_charged_mean"] - 1.0) < 1e-12
    # the charge: order minus baseline, each 1 when it uses the five-dual values
    assert (g.INTER_FIVE in g.USES_FIVE_DUALS) and (g.FIVE in g.USES_FIVE_DUALS) and g.JOINTDW not in g.USES_FIVE_DUALS


def test_single_prompt_group_uses_the_two_stage_bootstrap():
    from llmdw.scoring import cluster

    cluster.set_groups({f"i{k}": "g" for k in range(20)})
    try:
        # one group: stage 1 always draws it, stage 2 resamples the instances within it, so varying values have SE > 0
        _, _, se = cluster.boot({f"i{k}": float(k) for k in range(20)})
        assert se > 0.5
        # identical values (one request answered for every instance of the group) have SE 0 under any bootstrap
        _, _, se0 = cluster.boot({f"i{k}": 7.0 for k in range(20)})
        assert se0 < 1e-12
    finally:
        cluster.set_groups({})
