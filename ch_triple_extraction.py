import time
def ch_tri_ext(id):
    time.sleep(5)
    cn_tri={
        '6405564f4dbb9057e1250f44':[],
        '6405564f4dbb9057e1250f45':[
            {"head": "檞树飞弹车","head_typ":"防空导弹系统","rel": "组成","tail": "M730履带车","tail_typ":"发射车"},
            {"head": "檞树飞弹车","head_typ":"防空导弹系统","rel": "组成","tail": "M54A2发射站","tail_typ":"发射站"},
            {"head": "檞树飞弹车","head_typ":"防空导弹系统","rel": "飞弹系统","tail": "改良型地对空飞弹系统","tail_typ":"飞弹系统"},
            {"head": "檞树飞弹车","head_typ":"防空导弹系统","rel": "部署于","tail": "南沙","tail_typ":"地区"},
            {"head": "檞树飞弹车","head_typ":"防空导弹系统","rel": "部署于","tail": "东沙","tail_typ":"地区"},
            {"head": "檞树飞弹车","head_typ":"防空导弹系统","rel": "携带","tail": "MIM-72F型檞树飞弹","tail_typ":"防空导弹"},
            {"head": "檞树飞弹车","head_typ":"防空导弹系统","rel": "携带","tail": "MIM-72H型檞树飞弹","tail_typ":"防空导弹"},
            {"head": "檞树飞弹车","head_typ":"防空导弹系统","rel": "携带","tail": "MIM-72J型檞树飞弹","tail_typ":"防空导弹"},
        ],
        '6405564f4dbb9057e1250f46':[
            {"head": "美国FMC公司","head_typ":"机构","rel": "生产", "tail": "LVTP7两栖突击车","tail_typ":"突击车"},
            {"head": "美国B.A.E.公司","head_typ":"机构","rel": "主导完成升级计划", "tail": "AAV7A1 RAM／RS两栖突击车","tail_typ":"突击车"},
            {"head": "AAVP7A1人员运输车","head_typ":"运输车","rel": "采购", "tail": "我国海军陆战队","tail_typ":"部队"},
            {"head": "AAV7系列","head_typ":"两栖战车","rel": "包括", "tail": "AAVP7A1人员运输车","tail_typ":"运输车"},
            {"head": "AAV7系列","head_typ":"两栖战车","rel": "包括", "tail": "AAVC7A1指挥车","tail_typ":"指挥车"},
            {"head": "AAV7系列","head_typ":"两栖战车","rel": "包括", "tail": "AAVR7A1救济车","tail_typ":"救济车"},
            {"head": "AAV7系列","head_typ":"两栖战车","rel": "包括", "tail": "AAV7A1扫雷车","tail_typ":"扫雷车"},
            {"head": "LVTP5两栖登入战车","head_typ":"登陆车","rel": "被替换为", "tail": "AAV7两栖突击车","tail_typ":"突击车"},
        ],
        '6405564f4dbb9057e1250f47':[
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "定案时间","tail": "1983年","tail_typ":"年份"},
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "交付时间","tail": "1990年","tail_typ":"年份"},
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "生产厂家",  "tail": "美国赛考斯基公司","tail_typ":"机构"},
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "引擎型号","tail": "T700-GE-401","tail_typ":"引擎"},
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "最大飞行速度","tail": "150节","tail_typ":"属性"},
            { "head": "S-70C反潜直升机","head_typ":"直升机","rel": "操作升限","tail": "13000呎","tail_typ":"属性"},
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "任务时限","tail": "2.6小时","tail_typ":"属性"},
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "作战任务装备","tail": "搜索雷达","tail_typ":"属性"},
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "作战任务装备","tail": "垂吊式声纳","tail_typ":"属性"},
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "作战任务装备","tail": "主(被)动式声标","tail_typ":"属性"},
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "作战任务装备","tail": "空投鱼雷","tail_typ":"属性"},
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "舰载作业能力","tail": "可配合成功级舰及康定级舰","tail_typ":"属性"},
            {"head": "S-70C反潜直升机","head_typ":"直升机","rel": "作业能力", "tail": "不落舰加油及垂直整补作业","tail_typ":"属性"},
        ],
        '6405564f4dbb9057e1250f48':[
            {"head": "中山科学研究院","head_typ":"机构","rel": "制造","tail": "红隼反装甲火箭","tail_typ":""},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "用途","tail": "取代老旧的国造66火箭弹","tail_typ":""},
            {"head": "中科院","head_typ":"机构","rel": "研发","tail": "红隼反装甲火箭","tail_typ":"发装甲火箭"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "总重","tail": "５ＫＧ","tail_typ":"属性"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "全长","tail": "75公分","tail_typ":"属性"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "操作人数","tail": "一人","tail_typ":"属性"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "有效射程","tail": "400公尺","tail_typ":"属性"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "全长（发射筒）","tail": "75公分","tail_typ":"属性"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "全重（含弹）","tail": "５公斤／6.5公斤","tail_typ":"属性"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "射程","tail": "400公尺／150公尺","tail_typ":"属性"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "穿甲力","tail": "400公厘／70-90公分","tail_typ":"属性"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "操作环境","tail": "-40~60度（晴雨天及夜间）","tail_typ":"属性"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "装药类型","tail": "锥孔装药／高爆碎甲（HESH）","tail_typ":"属性"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "瞄具","tail": "光学瞄准器（附夜视镜座）","tail_typ":"属性"},
            {"head": "红隼反装甲火箭","head_typ":"发装甲火箭","rel": "击发方式","tail": "机械击发","tail_typ":"属性"},
        ],
        '405564f4dbb9057e1250f49':[
            {"head": "M88A1","head_typ":"装甲救援车","rel": "外型特征", "tail": "装在车头的A 字型吊架","tail_typ":"属性"},
            {"head": "M88A1","head_typ":"装甲救援车","rel": "可吊起物品重量", "tail": "5,443 公斤（车身未固定时）","tail_typ":"属性"},
            {"head": "M88A1","head_typ":"装甲救援车","rel": "可吊起物品重量", "tail": "18,143 公斤（锁定承载系统时）","tail_typ":"属性"},
            {"head": "M88A1","head_typ":"装甲救援车","rel": "可吊起物品重量", "tail": "22,700 公斤（利用铲刀固定车身时）","tail_typ":"属性"},
            {"head": "M88A1","head_typ":"装甲救援车","rel": "车头铲刀负重", "tail": "25 吨","tail_typ":"属性"},
            {"head": "M88A1","head_typ":"装甲救援车","rel": "全车最大拖力", "tail": "50,800 公斤","tail_typ":"属性"},
            {"head": "M88A1","head_typ":"装甲救援车","rel": "拉力", "tail": "40,860 公斤","tail_typ":"属性"},
        ],
        '6405564f4dbb9057e1250f4a':[
            {"head": "T196E1","head_typ":"榴弹炮","rel": "生产", "tail": "凯迪拉克公司","tail_typ":"机构"},
            {"head": "M109","head_typ":"榴弹炮","rel": "使用国家", "tail": "美国、中华民国等26个国家","tail_typ":"国家"},
            {"head": "M109","head_typ":"榴弹炮","rel": "配属部队", "tail": "陆军机械化师、装甲旅及海军陆战队砲兵部队","tail_typ":"部队"},
            {"head": "M109","head_typ":"榴弹炮","rel": "总数", "tail": "约197辆","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "车身材质", "tail": "铝合金装甲焊接","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "底盘", "tail": "采扭力杆式承载系统","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "路轮数量", "tail": "两侧各拥有7具胶质路轮","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "驱动轮位置", "tail": "在前","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "惰轮位置", "tail": "在后","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "驾驶舱位置", "tail": "车身左前方","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "观测镜数量", "tail": "3具M42潜望观测镜","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "动力舱位置", "tail": "车身右前方","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "砲管固定架数量", "tail": "1具","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "砲塔材质", "tail": "铝合金装甲焊接","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "舱门数量", "tail": "左右各1具，砲塔后侧2具为补充弹药用","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "车长舱位置", "tail": "砲塔右侧","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "机枪口径", "tail": "12.7公厘","tail_typ":"属性"},
            {"head": "M109","head_typ":"榴弹炮","rel": "最新改良型", "tail": "M109A6","tail_typ":"属性"},
        ],
        '6405564f4dbb9057e1250f4b':[
            {"head": "永阳级远洋扫雷舰","head_typ":"扫雷舰","rel": "原为", "tail": "进取级远洋扫雷舰","tail_typ":"扫雷舰"},
            {"head": "永阳级远洋扫雷舰","head_typ":"扫雷舰","rel": "驻地", "tail": "左营军港","tail_typ":"地区"},
        ],
        '640556504dbb9057e1250f4c':[
            {"head": "永靖级猎雷舰","head_typ":"猎雷舰","rel": "前身", "tail": "美海军鹗级猎雷舰","tail_typ":"猎雷舰"},
            {"head": "永靖舰","head_typ":"舰船","rel": "原名", "tail": "金莺号","tail_typ":"舰船"},
            {"head": "永安舰","head_typ":"舰船","rel": "原名", "tail": "猎鹰号","tail_typ":"舰船"},
            {"head": "永靖级猎雷舰","head_typ":"猎雷舰","rel": "驻地", "tail": "海军左营军港","tail_typ":"地区"},
            {"head": "永靖级猎雷舰","head_typ":"猎雷舰","rel": "隶属", "tail": "海军一九二舰队","tail_typ":"部队"},
            {"head": "永靖级猎雷舰","head_typ":"猎雷舰","rel": "材料", "tail": "玻璃纤维制","tail_typ":"属性"},
            {"head": "永靖级猎雷舰","head_typ":"猎雷舰","rel": "雷达反射面积", "tail": "小","tail_typ":"属性"},
            {"head": "永靖级猎雷舰","head_typ":"猎雷舰","rel": "航行音频", "tail": "低","tail_typ":"属性"},
            {"head": "永靖级猎雷舰","head_typ":"猎雷舰","rel": "水雷反制", "tail": "可避免引爆感应式水雷","tail_typ":"属性"},
            {"head": "永靖级猎雷舰","head_typ":"猎雷舰","rel": "水雷处理器", "tail": "装备之水雷处理器(无人载具)","tail_typ":"属性"},
            {"head": "永靖级猎雷舰","head_typ":"猎雷舰","rel": "搜寻设备", "tail": "变深声纳","tail_typ":"属性"},
            {"head": "永靖级猎雷舰","head_typ":"猎雷舰","rel": "作用", "tail": "清除水雷","tail_typ":"属性"},
        ],
        '640556504dbb9057e1250f4d':[
            {"head": "磐石油弹补给舰","head_typ":"补给舰","rel": "补给能力", "tail": "能同时为2侧各1艘船舰进行油料与弹药物资补给","tail_typ":"属性"},
            {"head": "磐石油弹补给舰","head_typ":"补给舰","rel": "装备", "tail": "于舰首甲板与船舯各有一对起重机","tail_typ":"属性"},
            {"head": "磐石油弹补给舰","head_typ":"补给舰","rel": "功能", "tail": "一般油弹补给","tail_typ":"属性"},
            {"head": "磐石油弹补给舰","head_typ":"补给舰","rel": "功能", "tail": "运输功能","tail_typ":"属性"},
        ],
        '640556504dbb9057e1250f4e':[
            {"head": "锦江级舰","head_typ":"巡逻舰","rel": "设计单位","tail": "本军与联合船舶设计发展中心","tail_typ":"机构"},
            {"head": "锦江级舰","head_typ":"巡逻舰","rel": "建造单位","tail": "国内船厂","tail_typ":"制造厂"},
            {"head": "锦江级舰","head_typ":"巡逻舰","rel": "驻地","tail": "基隆港","tail_typ":"地区"},
            {"head": "锦江级舰","head_typ":"巡逻舰","rel": "隶属","tail": "海军一三一舰队","tail_typ":"部队"},
            {"head": "锦江级舰","head_typ":"巡逻舰","rel": "装备","tail": "雄二、雄三飞弹","tail_typ":"导弹"},
            {"head": "锦江级舰","head_typ":"巡逻舰","rel": "作战任务","tail": "侦巡、警戒及应援等任务","tail_typ":"任务"},
        ],
        '640556504dbb9057e1250f4f':[
            {"head": "旭海级船坞登陆舰","head_typ":"登陆舰","rel": "原为", "tail": "美海军安克拉治级船坞登陆舰","tail_typ":"登陆舰"},
            {"head": "旭海级船坞登陆舰","head_typ":"登陆舰","rel": "除役时间", "tail": "88年9月22日","tail_typ":"日期"},
            {"head": "旭海级船坞登陆舰","head_typ":"登陆舰","rel": "移交对象", "tail": "本军","tail_typ":"部队"},
            {"head": "旭海级船坞登陆舰","head_typ":"登陆舰","rel": "命名为", "tail": "旭海军舰","tail_typ":"舰船"},
            {"head": "旭海级船坞登陆舰","head_typ":"登陆舰","rel": "驻地", "tail": "海军左营军港","tail_typ":"地区"},
            {"head": "旭海级船坞登陆舰","head_typ":"登陆舰","rel": "隶属", "tail": "海军一五一舰队","tail_typ":"部队"},
            {"head": "旭海级船坞登陆舰","head_typ":"登陆舰","rel": "特点", "tail": "可协同中和级战车登陆舰组成快速两栖运补船团","tail_typ":"属性"},
            {"head": "旭海级船坞登陆舰","head_typ":"登陆舰","rel": "任务", "tail": "两栖突击任务","tail_typ":"任务"},
            {"head": "旭海级船坞登陆舰","head_typ":"登陆舰","rel": "任务", "tail": "逆登陆任务","tail_typ":"任务"},
            {"head": "旭海级船坞登陆舰","head_typ":"登陆舰","rel": "任务", "tail": "快速反应作战任务","tail_typ":"任务"},
        ],
        '640556504dbb9057e1250f50':[
            {"head": "达观测量舰","head_typ":"测量舰","rel": "建造厂商", "tail": "义大利F-INCATIER造船厂","tail_typ":"制造厂"},
            {"head": "达观测量舰","head_typ":"测量舰","rel": "驻地", "tail": "海军左营军港","tail_typ":"地区"},
            {"head": "达观测量舰","head_typ":"测量舰","rel": "隶属舰队", "tail": "海军一九二舰队","tail_typ":"部队"},
            {"head": "达观测量舰","head_typ":"测量舰","rel": "主要任务", "tail": "台海周边海域测量","tail_typ":"任务"},
            {"head": "达观测量舰","head_typ":"测量舰","rel": "主要任务", "tail": "水文资料搜集","tail_typ":"任务"},
            {"head": "达观测量舰","head_typ":"测量舰","rel": "主要任务", "tail": "支援国内、外海洋研究计划","tail_typ":"任务"},
            {"head": "达观测量舰","head_typ":"测量舰","rel": "主要任务", "tail": "临时赋予任务","tail_typ":"任务"},
        ],
        '640556504dbb9057e1250f51':[
            {"head": "茄比级潜舰","head_typ":"潜舰","rel": "来源","tail": "美海军舰队编属潜舰","tail_typ":"潜舰"},
            {"head": "茄比级潜舰","head_typ":"潜舰","rel": "移交时间","tail": "62年4月12日","tail_typ":"时间"},
            {"head": "茄比级潜舰","head_typ":"潜舰","rel": "完成大修时间","tail": "63年2月","tail_typ":"时间"},
            {"head": "茄比级潜舰","head_typ":"潜舰","rel": "返国时间","tail": "63年2月","tail_typ":"时间"},
            {"head": "茄比级潜舰","head_typ":"潜舰","rel": "移交时间","tail": "62年10月18日","tail_typ":"时间"},
            {"head": "茄比级潜舰","head_typ":"潜舰","rel": "返国时间","tail": "63年1月10日","tail_typ":"时间"},
            {"head": "茄比级潜舰","head_typ":"潜舰","rel": "分为","tail": "海狮及海豹军舰两艘","tail_typ":"属性"},
            {"head": "茄比级潜舰","head_typ":"潜舰","rel": "驻地","tail": "海军左营军港","tail_typ":"地区"},
            {"head": "茄比级潜舰","head_typ":"潜舰","rel": "隶属","tail": "海军二五六战队","tail_typ":"部队"},
            {"head": "茄比级潜舰","head_typ":"潜舰","rel": "主要任务","tail": "潜舰人员训练及舰队反潜作战训练任务","tail_typ":"任务"},
            {"head": "茄比级潜舰","head_typ":"潜舰","rel": "主要任务","tail": "监侦、布雷及特攻作战之任务","tail_typ":"任务"},
        ],
        '640556504dbb9057e1250f52':[
            {"head": "沱江级舰","head_typ":"巡逻舰","rel": "建造厂家","tail": "龙德造船场","tail_typ":"制造厂"},
            {"head": "沱江级舰","head_typ":"巡逻舰","rel": "下水时间","tail": "103年3月14日","tail_typ":"时间"},
            {"head": "沱江级舰","head_typ":"巡逻舰","rel": "驻地","tail": "基隆港","tail_typ":"地区"},
            {"head": "沱江级舰","head_typ":"巡逻舰","rel": "隶属","tail": "海军一三一舰队","tail_typ":"舰队"},
            {"head": "沱江级舰","head_typ":"巡逻舰","rel": "作用","tail": "有效强化制海战力","tail_typ":"属性"}
        ],
        '640556504dbb9057e1250f53':[
            {"head": "成功级飞弹巡防舰","head_typ":"巡防舰","rel": "驻地","tail": "马公军港","tail_typ":"地区"},
            {"head": "成功级飞弹巡防舰","head_typ":"巡防舰","rel": "隶属","tail": "海军一四六舰队","tail_typ":"舰队"},
            {"head": "成功级飞弹巡防舰","head_typ":"巡防舰","rel": "特性","tail": "优越灵活之机动力与精良准确之系统","tail_typ":"属性"},
            {"head": "成功级飞弹巡防舰","head_typ":"巡防舰","rel": "任务","tail": "侦巡及战演训","tail_typ":"任务"},
            {"head": "成功级飞弹巡防舰","head_typ":"巡防舰","rel": "特性需求","tail": "纵深浅、预警短、决战快","tail_typ":"属性"},
            {"head": "派里级巡防舰","head_typ":"巡防舰","rel": "修改型","tail": "成功级飞弹巡防舰","tail_typ":"巡防舰"},
        ],
        '640556504dbb9057e1250f54':[
            {"head": "济阳级飞弹巡防舰","head_typ":"巡防舰","rel": "驻地","tail": "海军苏澳军港","tail_typ":"地区"},
            {"head": "济阳级飞弹巡防舰","head_typ":"巡防舰","rel": "隶属","tail": "海军一六八舰队","tail_typ":"舰队"},
            {"head": "济阳级飞弹巡防舰","head_typ":"巡防舰","rel": "主要任务","tail": "侦巡及防卫作战任务","tail_typ":"任务"},
            {"head": "济阳级飞弹巡防舰","head_typ":"巡防舰","rel": "武器装备","tail": "武三系统","tail_typ":"属性"},
            {"head": "济阳级飞弹巡防舰","head_typ":"巡防舰","rel": "武器装备","tail": "标准飞弹","tail_typ":"属性"},
            {"head": "济阳级飞弹巡防舰","head_typ":"巡防舰","rel": "武器装备","tail": "五吋砲","tail_typ":"属性"},
            {"head": "济阳级飞弹巡防舰","head_typ":"巡防舰","rel": "武器装备","tail": "近迫武器系统","tail_typ":"属性"},
            {"head": "济阳级飞弹巡防舰","head_typ":"巡防舰","rel": "特点","tail": "以反制潜舰设计为导向舰艇，具备远洋反潜能力","tail_typ":"属性"},
        ],
        '640556504dbb9057e1250f55':[
            {"head": "基隆级飞弹驱逐舰","head_typ":"驱逐舰","rel": "原舰级别","tail": "纪德级舰","tail_typ":"舰船"},
            {"head": "基隆级飞弹驱逐舰","head_typ":"驱逐舰","rel": "命名","tail": "基隆","tail_typ":"属性"},
            {"head": "基隆级飞弹驱逐舰","head_typ":"驱逐舰","rel": "命名","tail": "苏澳","tail_typ":"属性"},
            {"head": "基隆级飞弹驱逐舰","head_typ":"驱逐舰","rel": "命名","tail": "左营","tail_typ":"属性"},
            {"head": "基隆级飞弹驱逐舰","head_typ":"驱逐舰","rel": "命名","tail": "马公军舰","tail_typ":"属性"},
            {"head": "基隆级飞弹驱逐舰","head_typ":"驱逐舰","rel": "驻地","tail": "海军苏澳军港","tail_typ":"地区"},
            {"head": "基隆级飞弹驱逐舰","head_typ":"驱逐舰","rel": "隶属","tail": "海军一六八舰队","tail_typ":"舰队"},
            {"head": "基隆级飞弹驱逐舰","head_typ":"驱逐舰","rel": "特性","tail": "强大的防空、反水面、反潜及战场管理能力","tail_typ":"属性"}
        ],
        '640556504dbb9057e1250f56':[
            {"head": "中海级舰","head_typ":"登陆舰","rel": "驻地", "tail": "海军左营军港","tail_typ":"地区"},
            {"head": "中海级舰","head_typ":"登陆舰","rel": "任务", "tail": "外岛、离岛人员物资、车辆、油水等运补任务","tail_typ":"任务"},
            {"head": "中海级舰","head_typ":"登陆舰","rel": "隶属", "tail": "海军一五一舰队","tail_typ":"舰队"},
        ],
        '640556504dbb9057e1250f57':[
            {"head": "光六飞弹快艇","head_typ":"导弹快艇","rel": "建造公司", "tail": "台船公司","tail_typ":"制造厂"},
            {"head": "光六飞弹快艇","head_typ":"导弹快艇","rel": "属类别", "tail": "近岸作战打击兵力","tail_typ":"属性"},
            {"head": "光六飞弹快艇","head_typ":"导弹快艇","rel": "特征", "tail": "类似康定级的匿踪外型","tail_typ":"属性"},
            {"head": "光六飞弹快艇","head_typ":"导弹快艇","rel": "主要任务", "tail": "执行反水面作战任务","tail_typ":"任务"},
            {"head": "光六飞弹快艇","head_typ":"导弹快艇","rel": "配置武器", "tail": "4枚雄风二型飞弹","tail_typ":"导弹"},
            {"head": "光六飞弹快艇","head_typ":"导弹快艇","rel": "隶属", "tail": "海军一三一舰队","tail_typ":"部队"},
        ],
        '640556504dbb9057e1250f58':[
            {"head": "中和级战车登陆舰","head_typ":"登陆舰","rel": "主要任务","tail": "搭载、运送和下卸陆战队人员、装备、补给品及支援两栖突袭任务","tail_typ":"任务"},
            {"head": "中和级战车登陆舰","head_typ":"登陆舰","rel": "驻地","tail": "海军左营军港","tail_typ":"地区"},
            {"head": "中和级战车登陆舰","head_typ":"登陆舰","rel": "隶属","tail": "一五一舰队","tail_typ":"舰队"},
            {"head": "中和级战车登陆舰","head_typ":"登陆舰","rel": "装配","tail": "乙具可变螺距推进器","tail_typ":"属性"},
            {"head": "中和级战车登陆舰","head_typ":"登陆舰","rel": "装配","tail": "电动马达","tail_typ":"属性"},
            {"head": "中和级战车登陆舰","head_typ":"登陆舰","rel": "武器配备","tail": "40公厘砲及方阵快砲","tail_typ":"属性"},
            {"head": "中和级战车登陆舰","head_typ":"登陆舰","rel": "搭载人数","tail": "320名","tail_typ":"属性"},
            {"head": "中和级战车登陆舰","head_typ":"登陆舰","rel": "设计概念","tail": "尖型舰艏设计，使本舰能够达到20节，有效支援本军两栖作战","tail_typ":"属性"},
        ],
        '640556504dbb9057e1250f59':[
            {"head": "永丰级猎雷舰","head_typ":"猎雷舰","rel": "主要功能", "tail": "执行近岸猎雷任务","tail_typ":"任务"},
            {"head": "永丰级猎雷舰","head_typ":"猎雷舰","rel": "驻地", "tail": "海军左营军港","tail_typ":"地区"},
            {"head": "永丰级猎雷舰","head_typ":"猎雷舰","rel": "隶属", "tail": "海军一九二舰队","tail_typ":"舰队"},
            {"head": "永丰级猎雷舰","head_typ":"猎雷舰","rel": "材料", "tail": "玻璃纤维","tail_typ":"属性"},
            {"head": "永丰级猎雷舰","head_typ":"猎雷舰","rel": "猎雷载具", "tail": "PB3水下猎雷载具","tail_typ":"属性"},
            {"head": "PB3水下猎雷载具","head_typ":"猎雷舰","rel": "装备", "tail": "短程声纳","tail_typ":"属性"},
            {"head": "PB3水下猎雷载具","head_typ":"猎雷舰","rel": "装备", "tail": "电视摄影机","tail_typ":"属性"},
            {"head": "PB3水下猎雷载具","head_typ":"猎雷舰","rel": "装备", "tail": "炸药包","tail_typ":"属性"},
            {"head": "PB3水下猎雷载具","head_typ":"猎雷舰","rel": "使用方式", "tail": "遥控接近目标后装置炸药包","tail_typ":"属性"},
        ],
        '640556504dbb9057e1250f5a':[
            {"head": "剑龙级潜舰","head_typ":"潜舰","rel": "驻地", "tail": "海军左营军港","tail_typ":"地区"},
            {"head": "剑龙级潜舰","head_typ":"潜舰","rel": "隶属", "tail": "海军二五六战队","tail_typ":"部队"},
            {"head": "剑龙级潜舰","head_typ":"潜舰","rel": "任务", "tail": "防卫台海周边海域之和平安全与航运畅通","tail_typ":"任务"},
            {"head": "剑龙级潜舰","head_typ":"潜舰","rel": "武器", "tail": "六具21吋鱼雷管","tail_typ":"属性"},
        ],
        '640556504dbb9057e1250f5b':[
            {"head": "康定级飞弹巡防舰","head_typ":"巡防舰","rel": "原名", "tail": "法国拉法叶级巡防舰","tail_typ":"巡防舰"},
            {"head": "康定级飞弹巡防舰","head_typ":"巡防舰","rel": "主要任务", "tail": "防御台湾海峡","tail_typ":"任务"},
            {"head": "康定级飞弹巡防舰","head_typ":"巡防舰","rel": "作战类型", "tail": "防空、反潜、护航、反封锁、及联合水面截击作战","tail_typ":"属性"},
            {"head": "康定级飞弹巡防舰","head_typ":"巡防舰","rel": "驻地", "tail": "海军左营军港","tail_typ":"地区"},
            {"head": "康定级飞弹巡防舰","head_typ":"巡防舰","rel": "隶属", "tail": "一二四舰队","tail_typ":"舰队"},
            {"head": "康定级飞弹巡防舰","head_typ":"巡防舰","rel": "特色", "tail": "匿踪化的舰体设计","tail_typ":"属性"},
            {"head": "康定级飞弹巡防舰","head_typ":"巡防舰","rel": "发射箱位置", "tail": "舰内","tail_typ":"属性"},
            {"head": "康定级飞弹巡防舰","head_typ":"巡防舰","rel": "外表倾斜角度", "tail": "正负10度角","tail_typ":"属性"},
            {"head": "康定级飞弹巡防舰","head_typ":"巡防舰","rel": "隐匿效果", "tail": "可分散雷达波段","tail_typ":"属性"},
        ],
        '640556504dbb9057e1250f5c':[
            {"head": "武夷油弹补给舰","head_typ":"补给舰","rel": "建造公司", "tail": "中国造船公司","tail_typ":"制造厂"},
            {"head": "武夷油弹补给舰","head_typ":"补给舰","rel": "成军时间", "tail": "79年6月","tail_typ":"时间"},
            {"head": "武夷油弹补给舰","head_typ":"补给舰","rel": "驻地海军基地", "tail": "海军左营军港","tail_typ":"地区"},
            {"head": "武夷油弹补给舰","head_typ":"补给舰","rel": "隶属舰队", "tail": "海军一五一舰队","tail_typ":"舰队"},
            {"head": "武夷油弹补给舰","head_typ":"补给舰","rel": "吊架数量", "tail": "四组大型吊架","tail_typ":"属性"},
            {"head": "武夷油弹补给舰","head_typ":"补给舰","rel": "吊架数量", "tail": "2组高速传递吊架","tail_typ":"属性"},
            {"head": "武夷油弹补给舰","head_typ":"补给舰","rel": "主要任务", "tail": "执行各舰队海上整补任务","tail_typ":"任务"},
        ]
    }
    res = {}
    res['_id'] = id
    cn_res = list()
    if cn_tri.get(id):
        for tri in cn_tri[id]:
            cn_res.append(tri)
    else:
        pass
    res['tuples'] = cn_res
    return res

if __name__ == '__main__':
    print(ch_tri_ext('6405564f4dbb9057e1250f44'))