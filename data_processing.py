<<<<<<< HEAD
import re


###################################################################
'''
regular expression
< regular >
input: data
output: data in regular expression
'''
###################################################################


def regular(data):
    return [' '.join(re.sub('[-=+,#/$\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', '', sent.replace('_num_', ' ').replace('\n', ' ')).split()) for sent in data]
=======
import re


###################################################################
'''
regular expression
< regular >
input: data
output: data in regular expression
'''
###################################################################


def regular(data):
    return [' '.join(re.sub('[-=+,#/$\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', '', sent.replace('_num_', ' ').replace('\n', ' ')).split()) for sent in data]
>>>>>>> a276db7324688919f66b41a3801e2fc014889596
