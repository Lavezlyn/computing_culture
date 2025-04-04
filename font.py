import matplotlib.font_manager as fm

# 获取所有字体
fonts = fm.findSystemFonts()

# 过滤出中文字体
chinese_fonts = []
for font in fonts:
    try:
        # 获取字体属性
        prop = fm.FontProperties(fname=font)
        # 检查字体是否支持中文
        if 'CJK' in prop.get_name() or 'SC' in prop.get_name() or 'CN' in prop.get_name():
            chinese_fonts.append(prop.get_name())
    except:
        continue

# 去重并打印
chinese_fonts = list(set(chinese_fonts))
print("系统上可用的中文字体：")
for font in chinese_fonts:
    print(font)