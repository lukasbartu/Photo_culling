__author__ = 'Lukáš Bartůněk'

import pyexiv2
import operator
import json


def include_metadata_rating(img_list, q_file, t_a_ratio):
    t_a_ratio = t_a_ratio/100
    with open(q_file) as f:
        q_data = json.load(f)
    q_list = sorted(q_data, key=operator.itemgetter("id"))

    mixed_list = []
    highest_q = 0
    lowest_q = 100
    for q in q_list:
        quality = q["aesthetic_quality"] * (1-t_a_ratio) + q["technical_quality"] * t_a_ratio
        mixed_list.append(quality)
        if quality > highest_q:
            highest_q = quality
        if quality < lowest_q:
            lowest_q = quality

    interval_size = (highest_q - lowest_q) / 5
    for i, img in enumerate(img_list):
        rating = int(((mixed_list[i]-lowest_q)/interval_size)) + 1
        rating_percent = int(((mixed_list[i]-lowest_q) / (interval_size*5))*100)
        if rating == 6:
            rating = 5
        try:
            with pyexiv2.Image(img) as handle:
                meta = {'Exif.Image.Rating': rating,
                        'Exif.Image.RatingPercent': rating_percent}
                handle.modify_exif(meta)
        except Exception:
            raise Exception
