from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV = Path("data/e_ladelasten/EV_Lastprofile_gefiltert.csv")


def build_random_12h_lastprofile(
    csv_path: str | Path = DEFAULT_CSV,
    duration_hours: int = 12,
    random_state: int | None = None,
    first_excerpt_min_len: int = 3,
    first_excerpt_max_len: int | None = None,
) -> pd.DataFrame:
    """Baue ein zufälliges 12-Stunden-Lastprofil aus zufällig ausgewählten Profilen der CSV-Datei.

    Die Funktion zieht zunächst ein zufälliges Profil aus der CSV, wählt daraus einen
    zufälligen Ausschnitt ab dem 4. Wert (also nach den ersten 15 Minuten) und hängt
    anschließend weitere zufällig ausgewählte Profile an, bis genau 12 Stunden erreicht sind.

    Parameters
    ----------
    csv_path : str | Path
        Pfad zur CSV-Datei mit den Lastprofilen.
    duration_hours : int
        Ziel-Länge des erzeugten Profils in Stunden. Standard: 12.
    random_state : int | None
        Optionaler Seed für reproduzierbare Ergebnisse.
    first_excerpt_min_len : int
        Mindestlänge des ersten Ausschnitts in Werten.
    first_excerpt_max_len : int | None
        Maximale Länge des ersten Ausschnitts in Werten. Falls None, wird ein sinnvoller
        Standardwert genutzt.

    Returns
    -------
    pandas.DataFrame
        DataFrame mit den Spalten minute, lastprofil und quelle.
    """
    if random_state is not None:
        np.random.seed(random_state)

    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = (Path(__file__).resolve().parent / csv_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path)
    profile_cols = [col for col in df.columns if col != "minute"]
    if not profile_cols:
        raise ValueError("Keine Lastprofil-Spalten in der CSV gefunden.")

    data = {
        col: pd.to_numeric(df[col], errors="coerce").dropna().astype(float).to_numpy()
        for col in profile_cols
    }
    profile_cols = [col for col in profile_cols if len(data[col]) > 0]
    if not profile_cols:
        raise ValueError("Keine Lastprofil-Spalten mit echten Werten gefunden.")
    interval_minutes = 5
    total_points = int(duration_hours * 60 / interval_minutes)
    values: list[float] = []
    sources: list[str] = []
    remaining = total_points

    first_profile = np.random.choice(profile_cols)
    first_series = data[first_profile]
    if len(first_series) <= first_excerpt_min_len:
        first_start = 0
    else:
        first_start = np.random.randint(first_excerpt_min_len, len(first_series))
    max_len = len(first_series) - first_start
    if first_excerpt_max_len is None:
        first_excerpt_max_len = min(60, max_len)

    if remaining <= first_excerpt_min_len:
        first_excerpt_len = min(remaining, max_len)
    else:
        max_first_len = max(1, min(first_excerpt_max_len, max_len, remaining))
        if max_first_len < first_excerpt_min_len:
            first_excerpt_len = max_first_len
        else:
            first_excerpt_len = int(np.random.randint(first_excerpt_min_len, max_first_len + 1))
    first_excerpt_len = min(first_excerpt_len, remaining, max_len)

    first_chunk = first_series[first_start : first_start + first_excerpt_len]
    if len(first_chunk) == 0:
        raise ValueError("Der erste Ausschnitt war leer. Bitte die Daten prüfen.")
    values.extend(first_chunk.tolist())
    sources.extend([first_profile] * len(first_chunk))
    remaining -= len(first_chunk)

    while remaining > 0:
        profile_name = np.random.choice(profile_cols)
        profile_series = data[profile_name]
        if len(profile_series) == 0:
            break
        chunk_len = min(len(profile_series), remaining)
        chunk = profile_series[:chunk_len]
        if len(chunk) == 0:
            continue
        values.extend(chunk.tolist())
        sources.extend([profile_name] * len(chunk))
        remaining -= len(chunk)

    return pd.DataFrame(
        {
            "minute": np.arange(total_points) * interval_minutes,
            "lastprofil": values,
            "quelle": sources,
        }
    )


if __name__ == "__main__":
    profile = build_random_12h_lastprofile(random_state=42)
    print(profile.head())
    print(f"\nAnzahl Zeilen: {len(profile)}")
