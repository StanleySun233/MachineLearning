col = ["酒精", "苹果酸", "灰", "灰的碱性", "镁", "总酚", "类黄酮", "非黄烷类酚类", "花青素", "颜色强度", "色调", "稀释葡萄酒", "脯氨酸"]
cls = ["琴酒", "雪莉", "贝尔摩斯"]
X = pd.DataFrame(data=wine["data"], columns=col)
Y = pd.DataFrame(data=wine["target"], columns=["标签"])
df = pd.concat([X, Y], axis=1)
df.head()
