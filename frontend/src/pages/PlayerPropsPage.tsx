import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { PlayerPropPrediction } from "../types";

export function PlayerPropsPage({ season, week }: { season: number; week: number }) {
  const [props, setProps] = useState<PlayerPropPrediction[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // The rows came from an earlier snapshot build. Worth showing rather than
  // swallowing: a reader told "unavailable" yesterday and shown numbers today
  // deserves to know the numbers are older than the page looks.
  const [stale, setStale] = useState(false);
  const [sortBy, setSortBy] = useState<"anytime_td_prob" | "rushing_yards" | "receiving_yards" | "passing_yards">(
    "anytime_td_prob",
  );

  useEffect(() => {
    // `loading` is load-bearing, not decoration. Without a way to tell "still
    // fetching", "loaded and genuinely empty" and "the fetch failed" apart, all
    // three render as the same empty table -- which is how a backend 503 ends up
    // reading to a visitor as "no props for this game yet".
    setLoading(true);
    setError(null);
    setStale(false);
    api
      .playerProps(season, week)
      .then(({ data, stale: isStale }) => {
        setProps(data);
        setStale(isStale);
      })
      .catch((err) => {
        setError(err.message);
        setProps([]);
        setStale(false);
      })
      .finally(() => setLoading(false));
  }, [season, week]);

  const sorted = [...props].sort((a, b) => (b[sortBy] ?? 0) - (a[sortBy] ?? 0));

  return (
    <div>
      <h1>Player Props — Week {week}</h1>
      {loading && <p>Loading…</p>}
      {error && (
        <p role="alert">
          Player props could not be loaded: {error}
        </p>
      )}
      {!loading && !error && stale && (
        <p>
          These projections come from an earlier build &mdash; the latest rebuild of this
          week failed, so they may not reflect the current model.
        </p>
      )}
      {!loading && !error && !stale && props.length === 0 && (
        <p>No player projection props available for this week yet.</p>
      )}
      {!error && !loading && (
        <>
          <label>
            Sort by:{" "}
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value as typeof sortBy)}>
              <option value="anytime_td_prob">Anytime TD</option>
              <option value="passing_yards">Passing Yards</option>
              <option value="rushing_yards">Rushing Yards</option>
              <option value="receiving_yards">Receiving Yards</option>
            </select>
          </label>
          {/* The table is suppressed on failure, not just when there is nothing
              to show: a zero-row grid of column headers beside the error is a
              second rendering of "there are no props here", which is the same lie
              in a different font. */}
          <table>
            <thead>
              <tr>
                <th>Player</th>
                <th>Anytime TD</th>
                <th>Passing Yds</th>
                <th>Rushing Yds</th>
                <th>Receiving Yds</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((player) => (
                <tr key={player.player_id}>
                  <td>{player.player_name}</td>
                  <td>{Math.round(player.anytime_td_prob * 100)}%</td>
                  <td>{player.passing_yards != null ? Math.round(player.passing_yards) : "—"}</td>
                  <td>{player.rushing_yards != null ? Math.round(player.rushing_yards) : "—"}</td>
                  <td>{player.receiving_yards != null ? Math.round(player.receiving_yards) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
