# constants
scenes = {}
scenes["train"] = [
    'Allensville',
    'Beechwood',
    'Benevolence',
    'Coffeen',
    'Cosmos',
    'Forkland',
    'Hanson',
    'Hiteman',
    'Klickitat',
    'Lakeville',
    'Leonardo',
    'Lindenwood',
    'Marstons',
    'Merom',
    'Mifflinburg',
    'Newfields',
    'Onaga',
    'Pinesdale',
    'Pomaria',
    'Ranchester',
    'Shelbyville',
    'Stockman',
    'Tolstoy',
    'Wainscott',
    'Woodbine',
]

scenes["val"] = [
    'Collierville',
    'Corozal',
    'Darden',
    'Markleeville',
    'Wiconisco',
]

scenes["test"] = [
    'Collierville',
    'Corozal',
    'Darden',
    'Markleeville',
    'Wiconisco',
]

master_scene_dir = "/home/cbl/alp/data/scene_datasets/gibson_semantics/"

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    # "dining-table": 6,
    # "oven": 7,
    # "sink": 8,
    # "refrigerator": 9,
    # "book": 10,
    # "clock": 11,
    # "vase": 12,
    # "cup": 13,
    # "bottle": 14
}

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

action_mapping = {
    "move_forward": 0,
    "turn_left": 1,
    "turn_right": 2,
}
action_decode = {
    0: "move_forward",
    1: "turn_left",
    2: "turn_right",
}