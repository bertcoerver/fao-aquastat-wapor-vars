"""Registry describing the selectable spatial aggregation levels.

Each aggregation maps to a zipped ESRI shapefile and describes how to turn that
shapefile into one or more groupings: the `id_field` is burned into a raster and
used to group the pixels, the `name_field` labels the output rows, and each entry
in `groupings` is one set of polygons to reduce over (an optional attribute
`where` filter for rasterizing, plus the equivalent pandas `mask` used to pick
the expected group ids). Area is always computed from the geometry, since none of
these shapefiles ship an area attribute.
"""

# The legacy UN_AREAS.gpkg flagged only two areas as `is_cluster=1`: the European
# Union (m49=97) and China (m49=156), i.e. aggregate polygons that overlie their
# member territories. The new un_areas.zip drops the `is_cluster` column but still
# contains both m49 codes, so we reproduce the country/cluster split from this list.
CLUSTER_M49 = (97, 156)
_CLUSTER_SQL = "(" + ", ".join(str(c) for c in CLUSTER_M49) + ")"


AGGREGATIONS = {
    "country": {
        "url": "https://storage.googleapis.com/fao-maps-ckan-test/shp_file_direct_access/un_areas.zip",
        "id_field": "M49",
        "name_field": "NAME",
        "groupings": {
            "country": {
                "where": f"M49 NOT IN {_CLUSTER_SQL}",
                "mask": lambda gdf: ~gdf["M49"].isin(CLUSTER_M49),
            },
            "cluster": {
                "where": f"M49 IN {_CLUSTER_SQL}",
                "mask": lambda gdf: gdf["M49"].isin(CLUSTER_M49),
            },
        },
    },
    "basin": {
        "url": "https://storage.googleapis.com/fao-maps-catalog-data/projects/SDG6-Basin-Maps/FAO_MBM_v2.zip",
        "id_field": "FAO_MB_ID",
        "name_field": "S6_MB_NAME",
        "groupings": {"basin": {"where": None, "mask": None}},
    },
    "subbasin": {
        "url": "https://storage.googleapis.com/fao-maps-catalog-data/projects/SDG6-Basin-Maps/FAO_SBM_v2.zip",
        "id_field": "FAO_SB_ID",
        "name_field": "S6_SB_NAME",
        "groupings": {"subbasin": {"where": None, "mask": None}},
    },
}

# CLI choices, in a stable order.
AGGREGATION_CHOICES = list(AGGREGATIONS)
