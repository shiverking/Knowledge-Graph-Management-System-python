class Match(object):
    def __init__(self, stringA, stringB):
        if len(stringA) < len(stringB):
            self.min = stringA
            self.max = stringB
        else:
            self.max = stringA
            self.min = stringB

    def approximate_match(self):
        re = []
        desc = []
        for i in range(len(self.min)):

            for j in range(len(self.max)):

                r = self.min[i].find(self.max[j])  # 核心函数为find

                if r != -1:
                    re.append(r)
                    desc.append(self.max[j])

        match_ratio = len(re) / len(self.max)
        match_ratio = str(round(match_ratio * 100, 2))
        return match_ratio

    def exact_match(self):

        if self.a == self.b:
            return str(100)
        else:
            return str(0)


class Match_list(object):

    def __init__(self, item, data_list):
        self.item = item
        self.data_list = data_list

    def approximate_match(self, flag):
        item = self.item
        data_list = self.data_list
        temp = 0
        word = []
        for item2 in data_list:
            m = Match(str(item), str(item2))
            r = m.approximate_match()
            r = float(r)
            if(r>100):
                r=100
            if r >= float(temp) and r != 0 and r >= flag and r<100:
                temp = r
                word.append([item2,r/100])
        return word