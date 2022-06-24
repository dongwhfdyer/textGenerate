import jieba

sentence='搜索引擎技术的研究，国外比中国要早近十年，从最早的Archie，到后来的Excite，以及altvista、overture、google等搜索引擎面世，搜索引擎发展至今，已经有十几年的历史，而国内开始研究搜索引擎是在上世纪末本世纪初。在许多领域，都是国外的产品和技术一统天下，特别是当某种技术在国外研究多年而国内才开始的情况下。例如操作系统、字处理软件、浏览器等等，但搜索引擎却是个例外。虽然在国外搜索引擎技术早就开始研究，但在国内还是陆续涌现出优秀的搜索引擎，像百度（http://www.baidu.com）、和最近刚出的 有道（http://www.youdao.com）等。目前在中文搜索引擎领域，国内的搜索引擎已经和国外的搜索引擎效果上相差不远。但是SWJ认为其技术能力等方面还是相差国外先进水平有一定的距离 不过，这距离在慢慢的拉近中！ 谈到搜索引擎的分词技术之所以能形成现在这样的局面，有一个重要的原因就在于中文和英文两种语言自身的书写方式不同。'
jieba.cut(sentence)

print(list(jieba.cut(sentence, cut_all=False)))

print('3333333333333333333333333333333')

print(list(jieba.cut(sentence, cut_all=True)))

print('3333333333333333333333333333333')

print(list(jieba.cut_for_search(sentence)))



import hanlp
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库
HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。'])
