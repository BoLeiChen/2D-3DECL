import copy
import numpy as np

dataset = "mp3d"

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

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14
}



coco_index_mapping_array = [56,
                            57,
                            58,
                            59,
                            61,
                            62,
                            60,
                            69,
                            71,
                            72,
                            73,
                            74,
                            75,
                            41,
                            39]


coco_categories_mapping = {
    56: 0,  # chair
    59: 1,  # bed
    58: 2,  # potted plant
    61: 3,  # toilet
    62: 4,  # tv
    57: 5,  # couch
}



if dataset == "hm3d":
    color_palette = [
        1.0, 1.0, 1.0,
        0.6, 0.6, 0.6,
        0.95, 0.95, 0.95,
        0.96, 0.36, 0.26,
        0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
        0.68235294, 0.78039216, 0.90980392,
        0.59607843, 0.8745098, 0.54117647,
        0.17254902, 0.62745098, 0.17254902,
        1, 0.73333333, 0.47058824,
        0.7372549 , 0.74117647, 0.13333333,
        0.54901961, 0.3372549 , 0.29411765
        ]


    color_palette_array = np.asarray([
        [1.0, 1.0, 1.0],
        [0.6, 0.6, 0.6],
        [0.95, 0.95, 0.95],
        [0.96, 0.36, 0.26],
        [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
        [0.68235294, 0.78039216, 0.90980392],
        [0.59607843, 0.8745098, 0.54117647],
        [0.17254902, 0.62745098, 0.17254902],
        [1, 0.73333333, 0.47058824],
        [0.7372549 , 0.74117647, 0.13333333],
        [0.54901961, 0.3372549 , 0.29411765]])

if dataset == "mp3d":
    color_palette = [
        1.0, 1.0, 1.0,
        0.6, 0.6, 0.6,
        0.95, 0.95, 0.95,
        0.96, 0.36, 0.26,
        0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
        0.9400000000000001, 0.7818, 0.66,
        0.9400000000000001, 0.8868, 0.66,
        0.8882000000000001, 0.9400000000000001, 0.66,
        0.7832000000000001, 0.9400000000000001, 0.66,
        0.6782000000000001, 0.9400000000000001, 0.66,
        0.66, 0.9400000000000001, 0.7468000000000001,
        0.66, 0.9400000000000001, 0.8518000000000001,
        0.66, 0.9232, 0.9400000000000001,
        0.66, 0.8182, 0.9400000000000001,
        0.66, 0.7132, 0.9400000000000001,
        0.7117999999999999, 0.66, 0.9400000000000001,
        0.8168, 0.66, 0.9400000000000001,
        0.9218, 0.66, 0.9400000000000001,
        0.9400000000000001, 0.66, 0.8531999999999998,
        0.9400000000000001, 0.66, 0.748199999999999,
        0.300000000000001, 0.66, 0.8531999999999998,
        0.9400000000000001, 0.06, 0.8531999999999998,
        0.9400000000000001, 0.66, 0.5531999999999998,
        0.3400000000000001, 0.96, 0.2531999999999998,
        ]


    color_palette_array = np.asarray([
        [1.0, 1.0, 1.0],
        [0.6, 0.6, 0.6],
        [0.95, 0.95, 0.95],
        [0.96, 0.36, 0.26],
        [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
        [0.9400000000000001, 0.7818, 0.66],
        [0.9400000000000001, 0.8868, 0.66],
        [0.8882000000000001, 0.9400000000000001, 0.66],
        [0.7832000000000001, 0.9400000000000001, 0.66],
        [0.6782000000000001, 0.9400000000000001, 0.66],
        [0.66, 0.9400000000000001, 0.7468000000000001],
        [0.66, 0.9400000000000001, 0.8518000000000001],
        [0.66, 0.9232, 0.9400000000000001],
        [0.66, 0.8182, 0.9400000000000001],
        [0.66, 0.7132, 0.9400000000000001],
        [0.7117999999999999, 0.66, 0.9400000000000001],
        [0.8168, 0.66, 0.9400000000000001],
        [0.9218, 0.66, 0.9400000000000001],
        [0.9400000000000001, 0.66, 0.8531999999999998],
        [0.9400000000000001, 0.66, 0.748199999999999],
        [0.300000000000001, 0.66, 0.8531999999999998],
        [0.9400000000000001, 0.06, 0.8531999999999998],
        [0.9400000000000001, 0.66, 0.5531999999999998],
        [0.3400000000000001, 0.96, 0.2531999999999998]])


mpcat40_labels = [
    # '', # -1
    #'void', # 0
    'wall',
    'floor',
    'chair',
    'door',
    'table', # 5
    'picture',
    'cabinet',
    'cushion',
    'window',
    'sofa', # 10
    'bed',
    'curtain',
    'chest_of_drawers',
    'plant',
    'sink',
    'stairs',
    'ceiling',
    'toilet',
    'stool',
    'towel', # 20
    'mirror',
    'tv_monitor',
    'shower',
    'column',
    'bathtub',
    'counter',
    'fireplace',
    'lighting',
    'beam',
    'railing',
    'shelving',
    'blinds',
    'gym_equipment', # 33
    'seating',
    'board_panel',
    'furniture',
    'appliances',
    'clothes',
    'objects',
    'misc',
    'unlabeled' # 41
]




mp3d_habitat_labels = {
            'chair': 0, #g
            'table': 1, #g
            'picture':2, #b
            'cabinet':3, # in resnet
            'cushion':4, # in resnet
            'sofa':5, #g
            'bed':6, #g
            'chest_of_drawers':7, #b in resnet
            'plant':8, #g
            'sink':9, #g
            'toilet':10, #g
            'stool':11, #b
            'towel':12, #b in resnet
            'tv_monitor':13, #g
            'shower':14, #b
            'bathtub':15, #b in resnet
            'counter':16, #b isn't this table?
            'fireplace':17,
            'gym_equipment':18,
            'seating':19,
            'clothes':20, # in resnet
            'background': 21
}

hm3d_habitat_labels = {
            # 'background': 0,
            'chair': 0, #g
            'bed': 1, #g
            'plant':2, #b
            'toilet':3, # in resnet
            'tv_monitor':4, # in resnet
            'sofa':5,
            'background':6, #background
}



def get_habitat_labels(data_name):
    if data_name =="hm3d":
        return hm3d_habitat_labels
    elif data_name =="mp3d":
        return mp3d_habitat_labels

def get_fourty_dict(data_name):
    fourty2_dict = {}

    for i in range(len(mpcat40_labels)):
        lb = mpcat40_labels[i]
        if data_name =="hm3d":
            if lb in hm3d_habitat_labels.keys():
                fourty2_dict[i] = hm3d_habitat_labels[lb]
        elif data_name =="mp3d":
            if lb in mp3d_habitat_labels.keys():
                fourty2_dict[i] = mp3d_habitat_labels[lb]

    fourty2 = copy.deepcopy(fourty2_dict)
    return fourty2
