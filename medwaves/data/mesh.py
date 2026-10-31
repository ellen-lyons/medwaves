"""The mesh graph and its static node features (depth, distance to coast, directional fetch).

Node features never include absolute coordinates; lon/lat are kept in mesh.nc only for linking
observations, interpolating winds and drawing maps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from medwaves.data.common import chord_to_km, destination, to_xyz

DEPTH_RADIUS_KM = 10.0      # node depth = median sea depth within this radius
MIN_DEPTH_M = 5.0           # nodes on or next to land get this depth
FETCH_SHORE_KM = 30.0       # half a mesh spacing: how far a ray may cross land before reaching the sea


def read_gmsh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a gmsh 2.2 ASCII mesh: node lon, lat and 0-based triangles (n_tri, 3)."""
    lines = Path(path).read_text().splitlines()
    if lines[0] != "$MeshFormat" or not lines[1].startswith("2.2 0"):
        raise ValueError(f"{path}: expected gmsh 2.2 ASCII, got {lines[:2]}")
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    ids, lon, lat = np.empty(n, int), np.empty(n), np.empty(n)
    for k, line in enumerate(lines[i + 2:i + 2 + n]):
        parts = line.split()
        ids[k], lon[k], lat[k] = int(parts[0]), float(parts[1]), float(parts[2])
    index = {node_id: k for k, node_id in enumerate(ids)}
    j = lines.index("$Elements")
    tris = []
    for line in lines[j + 2:j + 2 + int(lines[j + 1])]:
        parts = line.split()
        if parts[1] == "2":                      # 3-node triangle; tags come before the nodes
            ntags = int(parts[2])
            tris.append([index[int(v)] for v in parts[3 + ntags:6 + ntags]])
    return lon, lat, np.asarray(tris, dtype=np.int64)


def edges_from_triangles(tri: np.ndarray) -> np.ndarray:
    """Directed edge index (2, E): every triangle side, in both directions, once."""
    pairs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    pairs = np.concatenate([pairs, pairs[:, ::-1]])
    return np.unique(pairs, axis=0).T


class Bathymetry:
    """ETOPO-style grid (altitude, positive up) with land mask and coastline lookups."""

    def __init__(self, path: Path):
        with xr.open_dataset(path) as ds:
            ds = ds.sortby("latitude").sortby("longitude")
            self.lat = ds["latitude"].values.astype(float)
            self.lon = ds["longitude"].values.astype(float)
            self.alt = ds["altitude"].values.astype(float)
        self.land = self.alt >= 0
        sea = ~self.land
        # Coastal land cells: land with a sea cell among its 4 neighbours.
        nb_sea = np.zeros_like(sea)
        nb_sea[1:, :] |= sea[:-1, :]
        nb_sea[:-1, :] |= sea[1:, :]
        nb_sea[:, 1:] |= sea[:, :-1]
        nb_sea[:, :-1] |= sea[:, 1:]
        ci, cj = np.nonzero(self.land & nb_sea)
        self._coast = cKDTree(to_xyz(self.lon[cj], self.lat[ci]))
        self.dlat = float(np.median(np.diff(self.lat)))
        self.dlon = float(np.median(np.diff(self.lon)))

    def _index(self, lon, lat):
        i = np.rint((np.asarray(lat) - self.lat[0]) / self.dlat).astype(np.int64)
        j = np.rint((np.asarray(lon) - self.lon[0]) / self.dlon).astype(np.int64)
        inside = (i >= 0) & (i < self.lat.size) & (j >= 0) & (j < self.lon.size)
        return np.clip(i, 0, self.lat.size - 1), np.clip(j, 0, self.lon.size - 1), inside

    def is_land(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """(land, inside_grid) for each point, nearest grid cell."""
        i, j, inside = self._index(lon, lat)
        return self.land[i, j] & inside, inside

    def distance_to_coast_km(self, lon, lat) -> np.ndarray:
        """Distance to the nearest coastal land cell; 0 for points on land."""
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        dist = chord_to_km(self._coast.query(to_xyz(lon, lat))[0]) if lon.size else np.zeros(0)
        land, _ = self.is_land(lon, lat)
        return np.where(land, 0.0, dist)

    def node_depth_m(self, lon, lat, radius_km: float = DEPTH_RADIUS_KM) -> np.ndarray:
        """Median sea depth (m, positive down) within radius_km of each point, at least MIN_DEPTH_M."""
        out = np.full(len(lon), MIN_DEPTH_M)
        di = int(np.ceil(radius_km / 111.2 / self.dlat))
        for k, (x, y) in enumerate(zip(lon, lat)):
            i, j, _ = self._index(x, y)
            dj = int(np.ceil(radius_km / (111.2 * max(np.cos(np.radians(y)), 0.1)) / self.dlon))
            sl = (slice(max(i - di, 0), i + di + 1), slice(max(j - dj, 0), j + dj + 1))
            glat, glon = np.meshgrid(self.lat[sl[0]], self.lon[sl[1]], indexing="ij")
            near = chord_to_km(np.linalg.norm(to_xyz(glon, glat) - to_xyz(x, y), axis=-1)) <= radius_km
            depths = -self.alt[sl][near & ~self.land[sl]]
            if depths.size:
                out[k] = max(float(np.median(depths)), MIN_DEPTH_M)
        return out

    def directional_fetch_km(self, lon, lat, n_dir: int, max_km: float, step_km: float,
                             shore_km: float = FETCH_SHORE_KM) -> np.ndarray:
        """Open-water distance from each point to land along n_dir great-circle bearings (0 = north).

        The mesh's boundary nodes sit on the coastline, often on a land cell, so a ray first walks
        off the shore: it starts at its first sea cell if that is within shore_km, and otherwise
        faces land (fetch 0). From there it runs to the next land cell. Rays that leave the grid,
        or reach max_km without land, get max_km.
        """
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        dist = np.concatenate([[0.0], np.arange(step_km, max_km + step_km / 2, step_km)])
        rows = np.arange(lon.size)
        out = np.zeros((lon.size, n_dir))
        for d in range(n_dir):
            plon, plat = destination(lon[:, None], lat[:, None], 360.0 * d / n_dir, dist[None, :])
            land, inside = self.is_land(plon, plat)
            sea = ~land & inside
            start = sea.argmax(axis=1)
            reaches_sea = sea.any(axis=1) & (dist[start] <= shore_km)
            after = np.arange(dist.size)[None, :] > start[:, None]
            stop = (land | ~inside) & after
            hit = stop.any(axis=1)
            first = stop.argmax(axis=1)
            fetch = np.where(hit & land[rows, first], dist[first] - dist[start], max_km)
            out[:, d] = np.where(reaches_sea, fetch, 0.0)
        return out


class Mesh:
    """Mesh nodes and triangles, with nearest-node and point-in-mesh lookups."""

    def __init__(self, lon: np.ndarray, lat: np.ndarray, triangles: np.ndarray):
        self.lon, self.lat, self.triangles = np.asarray(lon), np.asarray(lat), np.asarray(triangles)
        self.tree = cKDTree(to_xyz(self.lon, self.lat))
        degree = np.bincount(self.triangles.ravel(), minlength=self.lon.size)
        self.node_tris = np.full((self.lon.size, max(int(degree.max()), 1)), -1, np.int64)
        fill = np.zeros(self.lon.size, np.int64)
        for t, tri in enumerate(self.triangles):
            for v in tri:
                self.node_tris[v, fill[v]] = t
                fill[v] += 1

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Mesh:
        return cls(ds["lon"].values, ds["lat"].values, ds["triangles"].values)

    def nearest(self, lon, lat, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Indices (n, k) and great-circle distances in km (n, k) of the k nearest nodes."""
        chord, idx = self.tree.query(to_xyz(lon, lat), k=k)
        chord, idx = np.asarray(chord).reshape(-1, k), np.asarray(idx).reshape(-1, k)
        return idx.astype(np.int32), chord_to_km(chord).astype(np.float32)

    def covers(self, lon, lat, max_node_km: float) -> np.ndarray:
        """Inside a triangle, or within max_node_km of a node.

        The mesh's straight coastal edges cut corners, so many nearshore points lie just outside
        every triangle while still being well represented by a nearby node.
        """
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        if lon.size == 0:
            return np.zeros(0, bool)
        return self.contains(lon, lat) | (self.nearest(lon, lat, 1)[1][:, 0] <= max_node_km)

    def contains(self, lon, lat, chunk: int = 200_000) -> np.ndarray:
        """True for points inside a mesh triangle (lon/lat plane, fine at 60 km)."""
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        out = np.zeros(lon.size, bool)
        vx, vy = self.lon[self.triangles], self.lat[self.triangles]   # (n_tri, 3)
        for s in range(0, lon.size, chunk):
            x, y = lon[s:s + chunk], lat[s:s + chunk]
            idx, _ = self.nearest(x, y, k=min(6, self.lon.size))
            cand = self.node_tris[idx].reshape(len(x), -1)            # (n, 6 * max_degree)
            valid = cand >= 0
            c = np.where(valid, cand, 0)
            x0, y0 = vx[c, 0], vy[c, 0]
            x1, y1 = vx[c, 1], vy[c, 1]
            x2, y2 = vx[c, 2], vy[c, 2]
            det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            px, py = x[:, None], y[:, None]
            a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det
            b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det
            eps = -1e-9
            inside = (a >= eps) & (b >= eps) & (1 - a - b >= eps) & valid
            out[s:s + chunk] = inside.any(axis=1)
        return out


def build_mesh_dataset(msh_path: Path, bathy: Bathymetry, n_dir: int, fetch_max_km: float,
                       fetch_step_km: float) -> xr.Dataset:
    lon, lat, tri = read_gmsh(msh_path)
    edges = edges_from_triangles(tri)
    land, _ = bathy.is_land(lon, lat)
    elon, elat = lon[edges], lat[edges]
    edge_km = chord_to_km(np.linalg.norm(to_xyz(elon[0], elat[0]) - to_xyz(elon[1], elat[1]), axis=-1))
    ds = xr.Dataset(
        {
            "depth_m": ("node", bathy.node_depth_m(lon, lat).astype(np.float32)),
            "dist_coast_km": ("node", bathy.distance_to_coast_km(lon, lat).astype(np.float32)),
            "fetch_km": (("node", "direction"),
                         bathy.directional_fetch_km(lon, lat, n_dir, fetch_max_km, fetch_step_km)
                         .astype(np.float32)),
            "on_land": ("node", land),
            "triangles": (("triangle", "vertex"), tri.astype(np.int32)),
            "edge_index": (("pair", "edge"), edges.astype(np.int32)),
            "edge_km": ("edge", edge_km.astype(np.float32)),
        },
        coords={"lon": ("node", lon), "lat": ("node", lat),
                "direction": ("direction", 360.0 * np.arange(n_dir) / n_dir)},
        attrs={"source": str(msh_path.name),
               "node_features": "depth_m dist_coast_km fetch_km",
               "note": "lon/lat are for linking, interpolation and maps only, never model inputs"},
    )
    ds["direction"].attrs["units"] = "degrees clockwise from north (bearing the fetch is measured along)"
    return ds
