"""
駅周辺マクドナルド検索システム
R-Tree空間インデックスを使用した高速な店舗検索
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, render_template, request
from rtree import index

# 設定
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MC_PATH = os.path.join(APP_DIR, "Mc_six_fields_from_mc.json")
STATION_PATH = os.path.join(APP_DIR, "N02-24_Station.geojson")
app = Flask(__name__, template_folder=os.path.join(APP_DIR, "templates"))


# 地理計算関数
def haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine公式で2点間の距離を計算（メートル）"""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def meters_to_degree_buffer(lat: float, radius_m: float) -> Tuple[float, float]:
    """メートルを度数に変換（バウンディングボックス作成用）"""
    deg_per_meter_lat = 1.0 / 111320.0
    deg_per_meter_lon = 1.0 / (111320.0 * max(0.01, math.cos(math.radians(lat))))
    return radius_m * deg_per_meter_lon, radius_m * deg_per_meter_lat


# データ読み込み関数
def load_mc_points(path: str) -> List[Dict[str, Any]]:
    """マクドナルド店舗データを読み込む"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points: List[Dict[str, Any]] = []
    for obj in data:
        try:
            points.append({
                "id": int(obj["id"]) if str(obj.get("id", "")).isdigit() else obj.get("id"),
                "key": str(obj.get("key", "")),
                "name": str(obj.get("name", "")),
                "address": str(obj.get("address", "")),
                "lat": float(obj["latitude"]),
                "lon": float(obj["longitude"]),
            })
        except Exception:
            continue
    return points


def geometry_to_point(geom: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """GeoJSONジオメトリを点座標(経度, 緯度)に変換（非点タイプは重心を計算）"""
    gtype, coords = geom.get("type"), geom.get("coordinates")
    if not coords:
        return None
    
    # 点タイプ
    if gtype == "Point":
        try:
            return float(coords[0]), float(coords[1])
        except Exception:
            return None
    
    # 線・面タイプは重心を計算
    flat_coords = []
    if gtype == "LineString":
        flat_coords = coords
    elif gtype == "MultiLineString":
        flat_coords = [pt for line in coords for pt in line]
    elif gtype == "Polygon":
        flat_coords = [pt for ring in coords for pt in ring]
    elif gtype == "MultiPolygon":
        flat_coords = [pt for poly in coords for ring in poly for pt in ring]
    
    if not flat_coords:
        return None
    
    try:
        xs = [float(c[0]) for c in flat_coords]
        ys = [float(c[1]) for c in flat_coords]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    except Exception:
        return None


def load_stations(path: str) -> List[Dict[str, Any]]:
    """GeoJSONファイルから駅データを読み込む"""
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    stations: List[Dict[str, Any]] = []
    for feat in gj.get("features", []):
        pt = geometry_to_point(feat.get("geometry") or {})
        if pt is None:
            continue
        lon, lat = pt
        props = feat.get("properties") or {}
        name = props.get("N02_005") or props.get("name") or props.get("StationName") or ""
        stations.append({"name": str(name), "lat": lat, "lon": lon, "properties": props})
    return stations


# 検索アルゴリズム
def linear_search(points: List[Dict[str, Any]], lon: float, lat: float, radius_m: float) -> List[Dict[str, Any]]:
    """線形探索：全件走査で中心点から半径内の店舗を検索 (O(N))"""
    results: List[Dict[str, Any]] = []
    for pt in points:
        d = haversine_meters(lon, lat, pt["lon"], pt["lat"])
        if d <= radius_m:
            results.append({**pt, "distance_m": round(d, 2)})
    results.sort(key=lambda r: r["distance_m"])
    return results


class McRTree:
    """R-Tree空間インデックスを使用してマクドナルド店舗の周辺検索を高速化 (平均O(log N))"""
    
    def __init__(self, points: List[Dict[str, Any]]):
        """R-Treeインデックスを初期化"""
        p = index.Property()
        p.dimension = 2
        self._index = index.Index(properties=p)
        self._points = points
        for i, pt in enumerate(points):
            self._index.insert(i, (pt["lon"], pt["lat"], pt["lon"], pt["lat"]))

    def radius_search(self, lon: float, lat: float, radius_m: float) -> List[Dict[str, Any]]:
        """指定点周辺の半径内にあるマクドナルド店舗を検索"""
        dx, dy = meters_to_degree_buffer(lat, radius_m)
        bbox = (lon - dx, lat - dy, lon + dx, lat + dy)
        candidates = list(self._index.intersection(bbox))
        results: List[Dict[str, Any]] = []
        for idx_id in candidates:
            pt = self._points[idx_id]
            d = haversine_meters(lon, lat, pt["lon"], pt["lat"])
            if d <= radius_m:
                results.append({**pt, "distance_m": round(d, 2)})
        results.sort(key=lambda r: r["distance_m"])
        return results


# 起動時データ読み込み
MC_POINTS = load_mc_points(MC_PATH) if os.path.exists(MC_PATH) else []
STATIONS = load_stations(STATION_PATH) if os.path.exists(STATION_PATH) else []
MC_INDEX = McRTree(MC_POINTS) if MC_POINTS else None


# ルーティング
@app.route("/", methods=["GET"])
def index_page():
    """トップページ：検索フォームを表示"""
    return render_template("index.html", station_count=len(STATIONS), mc_count=len(MC_POINTS))


@app.route("/query", methods=["GET"])
def query():
    """検索処理：指定された駅周辺のマクドナルド店舗を検索"""
    radius_m_str = request.args.get("radius_m", "500").strip()
    station_query = request.args.get("station_query", "").strip()

    if not station_query:
        return render_template(
            "results.html",
            radius_m=500.0,
            results=[],
            stats=None,
            error="駅名を入力してください",
            station_query="",
            performance=None,
        )

    try:
        radius_m = float(radius_m_str)
        if radius_m <= 0:
            raise ValueError()
    except Exception:
        radius_m = 500.0

    if MC_INDEX is None or not STATIONS:
        return render_template(
            "results.html",
            radius_m=radius_m,
            results=[],
            stats=None,
            error="データが読み込まれていません。",
            station_query=station_query,
            performance=None,
        )

    # 駅名でフィルタリング
    st_list = [s for s in STATIONS if station_query in (s["name"] or "")]
    if not st_list:
        return render_template(
            "results.html",
            radius_m=radius_m,
            results=[],
            stats=None,
            error=f"「{station_query}」を含む駅名が見つかりませんでした",
            station_query=station_query,
            performance=None,
        )
    
    # パフォーマンス比較（最初の駅でテスト）
    performance_comparison = {
        "rtree_time_ms": 0.0,
        "linear_time_ms": 0.0,
        "total_points": len(MC_POINTS),
    }
    
    if st_list:
        test_station = st_list[0]
        test_lon, test_lat = test_station["lon"], test_station["lat"]
        
        start_time = time.perf_counter()
        MC_INDEX.radius_search(test_lon, test_lat, radius_m)
        performance_comparison["rtree_time_ms"] = round((time.perf_counter() - start_time) * 1000, 3)
        
        start_time = time.perf_counter()
        linear_search(MC_POINTS, test_lon, test_lat, radius_m)
        performance_comparison["linear_time_ms"] = round((time.perf_counter() - start_time) * 1000, 3)
    
    # 実際の検索処理
    results: List[Dict[str, Any]] = []
    for st in st_list:
        matches = MC_INDEX.radius_search(st["lon"], st["lat"], radius_m)
        results.append({
            "station_name": st["name"] or f"{st['lat']:.6f},{st['lon']:.6f}",
            "station_lat": st["lat"],
            "station_lon": st["lon"],
            "has_mc": len(matches) > 0,
            "matches": matches,
            "match_count": len(matches),
        })

    total_stations = len(results)
    stations_with_mc = sum(1 for r in results if r["has_mc"])
    stations_without_mc = total_stations - stations_with_mc
    total_mc_found = sum(len(r["matches"]) for r in results)
    
    stats = {
        "total_stations": total_stations,
        "stations_with_mc": stations_with_mc,
        "stations_without_mc": stations_without_mc,
        "total_mc_found": total_mc_found,
        "filtered": False,
    }

    return render_template(
        "results.html",
        radius_m=radius_m,
        results=results,
        stats=stats,
        error=None,
        station_query=station_query,
        performance=performance_comparison,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
