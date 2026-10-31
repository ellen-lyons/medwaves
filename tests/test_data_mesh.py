import numpy as np
import pytest

from medwaves.config import config_from_dict
from medwaves.data import synthetic
from medwaves.data.common import destination, haversine_km
from medwaves.data.mesh import Bathymetry, Mesh, build_mesh_dataset, edges_from_triangles, read_gmsh


@pytest.fixture
def files(raw_config, tmp_path):
    cfg = config_from_dict(raw_config, tmp_path)
    synthetic.write_mesh(tmp_path / "m.msh")
    synthetic.write_etopo(tmp_path / "etopo.nc", cfg)
    return tmp_path / "m.msh", Bathymetry(tmp_path / "etopo.nc")


def test_read_gmsh_and_edges(files):
    lon, lat, tri = read_gmsh(files[0])
    assert tri.min() == 0 and tri.max() == len(lon) - 1
    edges = edges_from_triangles(tri)
    pairs = set(map(tuple, edges.T))
    assert all((b, a) in pairs for a, b in pairs)                 # both directions
    assert len(pairs) == edges.shape[1]                           # no duplicates
    assert not any(a == b for a, b in pairs)


def test_destination_distance():
    lon2, lat2 = destination(5.0, 40.0, 90.0, 100.0)
    assert haversine_km(5.0, 40.0, lon2, lat2) == pytest.approx(100.0, rel=1e-6)


def test_fetch_short_towards_island_long_across_basin(files):
    _, bathy = files
    x, y = synthetic.ISLAND[0], synthetic.ISLAND[1] - 1.0          # 1 degree south of the island
    fetch = bathy.directional_fetch_km([x], [y], 16, 2000, 2)[0]
    north, south = fetch[0], fetch[8]
    assert north == pytest.approx((1.0 - synthetic.ISLAND[2]) * 111.2, abs=8)   # to the island shore
    assert south == pytest.approx((y - synthetic.SEA[2]) * 111.2, abs=8)        # to the basin edge
    assert (fetch <= 2000).all() and (fetch > 0).all()


def test_fetch_capped_when_ray_leaves_grid(raw_config, tmp_path):
    cfg = config_from_dict(raw_config, tmp_path)
    synthetic.write_etopo(tmp_path / "e.nc", cfg)
    bathy = Bathymetry(tmp_path / "e.nc")
    bathy.land[:] = False                                         # all sea: every ray leaves the grid
    assert (bathy.directional_fetch_km([5.0], [40.0], 16, 500, 2) == 500).all()


def test_distance_to_coast_and_depth(files):
    _, bathy = files
    d = bathy.distance_to_coast_km([5.0, 6.0, 0.0], [40.0, 40.0, 40.0])
    assert d[0] == pytest.approx((1.0 - synthetic.ISLAND[2]) * 111.2 * np.cos(np.radians(40)), abs=8)
    assert d[1] == 0 and d[2] == 0                                # island centre, mainland
    depth = bathy.node_depth_m([5.0, 6.0], [40.0, 40.0])
    assert depth[0] == 2000 and depth[1] == 5.0                   # land gets the minimum depth


def test_mesh_dataset_has_no_coordinate_features(files):
    msh, bathy = files
    ds = build_mesh_dataset(msh, bathy, 16, 2000, 2)
    features = ds.attrs["node_features"].split()
    assert set(features) == {"depth_m", "dist_coast_km", "fetch_km"}
    assert not {"lon", "lat"} & set(features)
    assert ds["fetch_km"].shape == (ds.sizes["node"], 16)
    assert ds["edge_index"].values.max() < ds.sizes["node"]


def test_contains(files):
    lon, lat, tri = read_gmsh(files[0])
    mesh = Mesh(lon, lat, tri)
    inside = mesh.contains([5.0, lon.min() - 0.1, 5.0, lon[0]], [40.0, 40.0, lat.max() + 0.2, lat[0]])
    assert inside.tolist() == [True, False, False, True]          # a mesh vertex counts as inside
    idx, dist = mesh.nearest([lon[3]], [lat[3]], 3)
    assert idx[0, 0] == 3 and dist[0, 0] == pytest.approx(0, abs=1e-6)
    assert (np.diff(dist[0]) >= 0).all()


def test_fetch_from_a_shoreline_node(files):
    _, bathy = files
    x, y, r = synthetic.ISLAND
    on_shore = y - r + 0.03                                       # on the island, ~3 km from its south shore
    fetch = bathy.directional_fetch_km([x, x], [on_shore, y], 16, 2000, 2)
    assert bathy.is_land([x], [on_shore])[0][0]
    assert fetch[0, 8] == pytest.approx((on_shore - 0.03 - synthetic.SEA[2]) * 111.2, abs=10)  # south: open
    assert fetch[0, 0] == 0                                       # north: across the island, faces land
    assert (fetch[1] == 0).all()                                  # island centre: land in every direction


def test_covers_nearshore_points_outside_triangles(files):
    lon, lat, tri = read_gmsh(files[0])
    mesh = Mesh(lon, lat, tri)
    x, y = [lon.min() - 0.2, lon.min() - 1.0], [lat.mean(), lat.mean()]   # ~17 km and ~85 km outside
    assert not mesh.contains(x, y).any()
    assert mesh.covers(x, y, 30).tolist() == [True, False]
