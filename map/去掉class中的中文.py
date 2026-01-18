# -*- coding: utf-8 -*-
import geopandas as gpd

shp_path = r"D:\map2\世界底图\Koppen_1991_2020_NA_big7.shp"
gdf = gpd.read_file(shp_path)

mapping = {
    "A 热带": "A",
    "B 干旱": "B",
    "C 温带": "C",
    "Dfa 冷带全湿炎夏": "Dfa",
    "Dfb 冷带全湿暖夏": "Dfb",
    "D 其他冷带": "D other",
    "E 极地": "E"
}

gdf["class"] = gdf["class"].replace(mapping)

out_path = r"D:\map2\世界底图\Koppen_1991_2020_NA_big7_simple.shp"
gdf.to_file(out_path, encoding="utf-8")

print("修改完成，保存到：", out_path)
print(gdf[["code", "class"]].drop_duplicates())
