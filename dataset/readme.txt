评测数据集说明
通用微博训练数据集包括27,768条微博，验证集（实时刷榜验证集）包含2,000条微博，测试数据集(最终评测集)包含5,000条微博。
疫情微博训练数据集包括8,606条微博，验证集（实时刷榜验证集）包含2,000条微博，测试数据集(最终评测集)包含3,000条微博。
注意：实际发布的测试集中会包含混淆数据（两个评测数据集均混淆至50000条），混淆数据不作为测点，在最终结果评测时会预先去除。


每个数据集提供txt和xlsx（WPS打开）两种版本，内容相同，仅格式不同。

usual_XX.txt表示通用微博数据集，virus_XX.txt表示疫情微博数据集。
XX_train.txt表示训练集。
XX_eval.txt表示验证集，是评测期间刷榜测试集。
XX_test.txt表示测试集，即最终评测集（分为含混淆数据版本和真实评测数据版本）。

XX_XX_labeled.txt表示数据集XX_XX.txt的有标签版本，在包含混淆数据的文件中，混淆数据的标签为None。


txt文件为json格式，格式样例如下所示：
[
	{
		"id": 1,
		"content": "回忆起老爸的点点滴滴，心痛…为什么.接受不了", 
		"label": "angry"
	},
	{
		"id": 2,
		"content": "我竟然不知道kkw是丑女无敌里的那个", 
		"label": "happy"
	},
	{
		"id": 3,
		"content": "我们做不到选择缘分，却可以珍惜缘分。", 
		"label": "neutral"
	}
]
现在的label有happy、sad、angry、neutral、surprise、fear
相对应的，xlsx中的第一列为id，第二列为content，第三列为label（仅训练数据和XX_labeled.xlsx有label列）。

评测通知网站：https://smp2020ewect.github.io/
评测提交排行网：http://39.97.118.137/
最终排行榜：http://39.97.118.137/test_rank
