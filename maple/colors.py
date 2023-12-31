import matplotlib.pyplot as plt

kelly = dict(vivid_yellow=(255, 179, 0),
             strong_purple=(128, 62, 117),
             vivid_orange=(255, 104, 0),
             very_light_blue=(166, 189, 215),
             vivid_red=(193, 0, 32),
             grayish_yellow=(206, 162, 98),
             medium_gray=(129, 112, 102),

             # these aren't good for people with defective color vision:
             vivid_green=(0, 125, 52),
             strong_purplish_pink=(246, 118, 142),
             strong_blue=(0, 83, 138),
             strong_yellowish_pink=(255, 122, 92),
             strong_violet=(83, 55, 122),
             vivid_orange_yellow=(255, 142, 0),
             strong_purplish_red=(179, 40, 81),
             vivid_greenish_yellow=(244, 200, 0),
             strong_reddish_brown=(127, 24, 13),
             vivid_yellowish_green=(147, 170, 0),
             deep_yellowish_brown=(89, 51, 21),
             vivid_reddish_orange=(241, 58, 19),
             dark_olive_green=(35, 44, 22))

extended_colortable = \
    plt.cm.tab10.colors + \
    plt.cm.Dark2.colors + \
    plt.cm.Accent.colors + \
    plt.cm.Set2.colors
