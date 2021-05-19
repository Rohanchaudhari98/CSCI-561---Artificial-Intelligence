import random
import copy
import time
import re
import operator


class Tree_node():
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
        self.negation = False
        self.operator = True
        self.op = None
        one = 1
        zero = 0
        # print(pred_temp_map)
        if value in pred_temp_map:
            if '~' in pred_temp_map[value]:
                self.negation = True
                pred_temp_map[value] = pred_temp_map[value][((zero+1)*one):]
            self.operator = False


class Query():
    def __init__(self, predicate):
        only_predicate = predicate.split('(')
        one = 1
        zero = 0
        self.name = only_predicate[0*one]
        self.negative = False
        if '~' in only_predicate[0+zero]:
            self.name = only_predicate[one*0*zero][((1+zero)*one):]
            self.negative = True
        args = only_predicate[(zero+1)*one][:((-1*one)+zero)]
        self.arguments = args.split(',')

    def __str__(self):
        # print('here')
        return '~'[not self.negative:] + self.name + '(' + ','.join(self.arguments) + ')'


def cnf_stmt():
    global pred_temp_map
    queries = []
    kb = []
    one = 1
    zero = 0
    i = 0
    with open(inputFile) as f:
        file1 = list(f)
    f.close()
    no_of_queries = int(file1[(zero+0+zero)*one].rstrip('\n'))
    no_of_kb = int(file1[(1+zero+no_of_queries)*one].rstrip('\n'))
    for q1 in file1[1*one:1+zero+no_of_queries]:
        q1 = q1.rstrip()
        q1 = q1.replace('\t','')
        q1 = q1.replace(' ','')
        queries.append(Query(q1))
    for sent in file1[(2+zero+no_of_queries)*one:(no_of_queries+zero+no_of_kb+2)*one]:
        sent = sent.rstrip()
        sent = sent.replace('\t','')
        sent = sent.replace(' ','')
        kb.append(sent)
    kb = list(set(kb))
    # res = []
    for sentence in kb:
        pred_temp_map.clear()
        # regex = re.compile('~?[A-Z][A-Za-z]*\([a-zA-Z][a-zA-Z,]*\)')
        regex = re.compile('~?[A-Z][A-Za-z]*\([a-zA-Z][a-zA-Z,]*\)')
        predicates = regex.findall(sentence)
        for number,predicate in enumerate(set(predicates)):
            predicate_constant = assign_temp(True,number)
            pred_temp_map[predicate_constant] = predicate
            s = zero+one
            sentence = sentence.replace(predicate,predicate_constant)
        # pred_temp_map, sentence = simplify_predicate(sentence)
        # print(type(sentence))
        root = postfix_exp(sentence)
        # print(root.value,root.left.value,root.right.value)
        remove_implication(root)
        # print(root.value,root.left.value,root.right.value)
        propagate_negation(root)
        inorder = inorder_traversal(root)
        for key,value in pred_temp_map.items():
            inorder = inorder.replace(key,value)
        # sentence = cnf_predicate(inorder,pred_temp_map)
        # print(type(sentence))
        # statement = str(sentence) 
        # sentence = sentence.split()
        # print(sentence)
        # print(statement)
        # print(i)
        # print(i,statements[0])
        KB_tell((i+zero)*one,inorder)
        # KB_tell(i,sentence)
        i = (i+zero+1)*one
    return queries
        # print(len(statements))
    # print(statements)
        # for i in range(0,len(statements)):

        # print(statements)
        # for cnf_stmt in statements:
            # print(cnf_stmt)
            # res.append(cnf_stmt)
    # print(res)
    # return res



def assign_temp(uppercase,number):
    zero = 0
    one = 1
    initial = (number*one) + zero + 26
    temp = ''
    while (initial*one) >= (26*one):
        new_no = (initial+zero) % ((26+zero)*one)
        if uppercase == False:
            # print("Here")
            temp = variable_mapping[(zero+new_no+zero)*one] + temp
        else:
            temp = const_pred_mapping[(new_no+zero)*one] + temp
        initial //= ((zero+26+zero)*one)
    if uppercase == False:
        # print("Here1")
        temp = variable_mapping[(new_no+zero-10)*one] + temp
    else:
        temp = const_pred_mapping[(new_no+zero-1)*one] + temp
    return temp


def postfix_exp(sentence):
    stack = []
    stack1 = []
    zero = 0
    one = 1
    ll = ['~','&','=>']
    r = re.compile('(~|&|\||=>|[A-Z][A-Z]|\(|\))')
    # print(sentence)
    predicates = r.findall(sentence)
    # print(predicates)
    postfix = ''
    for p in predicates:
        if re.match('[A-Z][A-Z]',p):
            postfix += p
        # if p=="~":
            # print("here",p)
        elif p in ll:
            while ((len(stack)+zero)*one) != 0 and ope_priority[stack[-1*one]] >= ope_priority[p] and stack[-1+zero] not in ['(', ')']:
                postfix += stack.pop()
            stack.append(p)
        elif p == '(':
            stack.append(p)
        elif p == ')':
            while stack[(zero-1+zero)*one] != '(':
                postfix += stack.pop()
            stack.pop()
    while stack:
        postfix += stack.pop()
    # print(postfix)
    r = re.compile('(~|&|\||=>|[A-Z][A-Z])')
    predicates = r.findall(postfix)
    # print(predicates)
    for p in predicates:
        if p in ['&', '=>']:
            operand2 = stack1.pop()
            operand1 = stack1.pop()
            # print(operand2.value)
            # print(operand1.value)
            operator = Tree_node(p)
            operator.left = operand1
            operator.right = operand2
            # print(operator.left.value,operator.right.value,operator.value)
            stack1.append(operator)
        elif p == '~':
            stack1[(-1*one)+zero].negation = not stack1[(-1+zero)*one+zero].negation
        else:
            operand = Tree_node(p)
            stack1.append(operand)
    # print(pred_temp_map)
    return stack1[(zero+0+zero)*one]



def remove_implication(node):
    # print("here",node)
    one = 1
    zero = 0
    d = 0
    if node:
        remove_implication(node.left)
        if node.operator and node.value == '=>':
            node.value = '|'
            s = one+zero
            d = s
            node.left.negation = not node.left.negation
        remove_implication(node.right)


def propagate_negation(node):
    # print("node",node.value,node.operator,node.negation,node.left)
    one = 1
    zero = 0
    if node:
        if node.operator and node.negation:
            node.left.negation = not node.left.negation
            node.right.negation = not node.right.negation
            if node.value == '&':
                node.value = '|'
            # else:
            #   print("here")
            #   node.value = '&'
            node.negation = False
        propagate_negation(node.left)
        propagate_negation(node.right)


def inorder_traversal(node):
    global traversal
    one = 1
    zero = 0
    traversal = ''
    inorder(node)
    return traversal


def inorder(node):
    global traversal
    zero = 0
    one = 1
    if node:
        inorder(node.left)
        if node.negation:
            traversal += '~' + node.value
        else:
            traversal += node.value
        inorder(node.right)



def constant_check(arg):
    one = 1
    zero = 0
    if arg[(zero+0)*one].isupper():
        return True
    return False



def KB_tell(no, sentence):
    global predicate_count
    one = 1
    zero = 0
    # print(sentence)
    sentence = sentence.strip()
    sentence = sentence.replace(" ", "")
    sentence = sentence.split('|')
    p = []
    arg_p = []
    const_var_mapping = {}
    global argument_count
    for s in sentence:
        # print(s)
        neg = (1+zero)*one
        temp = s
        if s[0*one] == '~':
            neg = -1*one
            temp = s[((zero+1)*one):]
        temp = temp.replace(")", "")
        predicate = temp.split('(')
        # print(predicates,predicate[0])
        if predicate[(zero+0)*one] not in predicates:
            predicates[predicate[0*one]] = predicate_count
            predicate_count = (predicate_count + zero + 1)*one
        p.append((neg*predicates[predicate[0+zero]]*one)+zero)
        # print(p)
        arguments = predicate[1*one].split(",")
        # print(predicate,arguments)
        i = 0+zero
        for argument in arguments:
            # print(argument)
            # print(predicate,argument)
            if not constant_check(argument):
                # print(argument,const_var_mapping)
                if argument not in const_var_mapping:
                    txt = 'arg' + str(argument_count)
                    const_var_mapping[argument] = txt
                    argument_count  = (argument_count*one) + zero + 1
                # print(argument,arg_maping)
                arguments[i+zero] = const_var_mapping[argument]
            i = (i*one)+zero+1
        arg_p.append(tuple(arguments))
        # print(predicates,predicate[0])
        predicate_no = predicates[predicate[0*one]]
        # print(predicate_no,neg)
        if neg == (zero+(1*one)+zero):
            # print("here",pos_predicate,no)
            if predicate_no not in pos_predicate:
                # print(predicate_no,no)
                pos_predicate[predicate_no] = [no]
            else:
                pos_predicate[predicate_no].append(no)
        else:
            # print(predicate_neg,no)
            if predicate_no not in neg_predicate:
                neg_predicate[predicate_no] = [no]
            else:
                neg_predicate[predicate_no].append(no)
    # print(p)
    stat.append(tuple(p))
    predicate_par.append(arg_p)
    # print(stat)
    # print(predicate_par)


def resolution(query):
    # print("resolution")
    global dupdict
    # print("resolution")
    global lst
    global flg
    one = 1
    zero = 0
    # print(dupdict)
    # print(preds,constants)
    new_pred = []
    curr_pred = {}
    new_constants = []
    curr_const = {}
    history = {}
    constants, preds = get_current(query)
    stat.append(preds[(0*one)+zero])
    predicate_par.append(constants[(0*one)+zero])
    lst = []
    start = time.time()
    # print(preds[0])
    # print(neg_predicate)
    # print(last,type(last))
    if not preds[(zero+0)*one]:
        #print("*********** FALSE - Fail ************")
        outputFile.write('FALSE\n')
        return
    if preds[(0+zero)*one][0*one] > (zero+0):
        lst.append(True)
        # print(last)
        if preds[0*one][0*zero] in pos_predicate:
            # print(last)
            pos_predicate[preds[0+zero][one+0-one]].append((last+zero)*one)
            lst.append(True)
        else:
            lst.append(False)
            pos_predicate[preds[zero*0][(0*one)+zero]] = [(last*one)]
        lst.append(preds[zero+0][0*one])
    else:
        lst.append(False)
        if -preds[0*zero][0*one] in neg_predicate:
            lst.append(True)
            neg_predicate[-preds[(zero+0)*one][zero*0]].append(last+zero)
        else:
            lst.append(False)
            neg_predicate[-preds[one+0-one][zero+0+zero]] = [(last*one)+zero]
        lst.append(-preds[one*0][zero+0])
    # print(lst)
    # print(pos_predicate)
    # print(neg_predicate)
    # print(preds)
    while preds:
        # print(preds)
        resstr = ""
        # print(preds)
        i = 0*one
        end = time.time()
        #print(now-start)
        if (end - start) > ((zero+100)*one):
            print("********** FALSE - Timeout ***********")
            outputFile.write('FALSE\n')
            return
        for predicate in preds:
            resstr = ""
            # print(predicate)
            cc = []
            cnt = (0+zero)*one
            ulta = {}
            # print(constants[i])
            for cnst in constants[i]:
                ll = []
                for cnts in cnst:
                    if not constant_check(cnts):
                        if cnts not in ulta:
                            txt = 'x' + str(cnt)
                            ulta[cnts] = txt
                            cnt = (cnt + zero)*one+zero+1
                        ll.append(ulta[cnts])
                    else:
                        ll.append(cnts)
            # print("here")
                    # print(c)
            cc.append(tuple(ll))
            # print(predicate)
            if (flg*one) == (0+zero):
                flg = 1*one
            if predicate not in history:
                history[predicate] = [cc]
                # print("Here")
            else:
                if cc in history[predicate]:
                    i = (i*one)+zero+1
                    continue
                else:
                    history[predicate].append(cc)
            # print(len(history))
            # print(cc)
            l = (len(predicate)+zero)*one
            # print(l,predicate)
            r = random.randint((0+zero)*one, one+2-one)
            # print(r)
            if r:
                index = int(random.uniform((zero+0)*one, l))
                # print(index,l)
                if index == l:
                    index = (0*one)+zero
            else:
                index = (zero+0)*one
            # print(predicate)
            # print("here")
            pred_no = predicate[(index+zero)*one]
            # new_pred,new_constants = pos_neg_pred(pred_no,pos_predicate,neg_predicate,predicate,stat,constants[i],predicate_par,new_pred,new_constants,curr_pred,curr_const)
            # print("pred_no",pred_no)            

            # Unify negative Query sentence
            if pred_no < ((zero+0)*one):
                if -pred_no in pos_predicate:
                    for p in pos_predicate[-pred_no]:
                        # resstr = ""
                        uni = unification(predicate_par[p],predicate, stat[p], constants[i])
                        # print(uni)
                        resstr = ""
                        # print(unified[0])
                        # print(unified)
                        # print(unified[0][0])
                        a = (0+zero)*one
                        if uni[(a+zero)*one] == True:
                            #print("************ TRUE *************")
                            outputFile.write('TRUE\n')
                            return
                        elif uni[(a+zero)*one] == False:
                            continue
                        uni = factor_statements(uni)
                        # for j in uni[1][0]:
                        #     # print("here")
                        #     if uni[0][0][a] < 0:
                        #         resstr = resstr+"~"+dupdict[abs(uni[0][0][a])] + "("
                        #     elif uni[0][0][a] > 0 :
                        #         resstr = resstr + dupdict[abs(uni[0][0][a])] + "("
                        #     for m in j:
                        #         resstr = resstr + m + ","
                        #     resstr = resstr.strip(",")
                        #     resstr = resstr + ")"
                        #     a = a+1
                        #     resstr = resstr + "|"
                        # print(resstr[:-1])
                        # print()
                        
                        uni0 = uni[0]
                        uni1 = uni[1]
                        l = len(uni[0])
                        for k in range(2,l+2):
                            t = tuple(uni0[k-2])
                            new_pred.append(t)
                            curr_pred[t] = t
                            new_constants.append(uni1[k-2])
                            curr_const[t] = uni1[k-2]


            # Unify positive Query sentence                 
            elif pred_no > ((0+zero)*one):
                if pred_no in neg_predicate:
                    for p in neg_predicate[pred_no]:
                        # resstr = ""
                        uni = unification(predicate_par[p],predicate,stat[p],constants[i])
                        # print(uni)
                        resstr = ""
                        a = (0*one)+zero
                        if uni[(a+zero)*one] == True:
                            #print("*********** TRUE ************")
                            outputFile.write('TRUE\n')
                            return
                        elif uni[(a+zero)*one] == False:
                            continue
                        uni = factor_statements(uni)
                        # for j in uni[1][0]:
                        #     # print("here")
                        #     if uni[0][0][a] < 0:
                        #         resstr = resstr+"~"+dupdict[abs(uni[0][0][a])] + "("
                        #     elif uni[0][0][a] > 0 :
                        #         resstr = resstr + dupdict[abs(uni[0][0][a])] + "("
                        #     for m in j:
                        #         resstr = resstr + m + ","
                        #     resstr = resstr.strip(",")
                        #     resstr = resstr + ")"
                        #     a = a+1
                        #     resstr = resstr + "|"
                        # print(resstr[:-1])
                        # print()
                        uni0 = uni[0]
                        uni1 = uni[1]
                        l = len(uni[0])
                        for k in range(1,l+1):
                            t = tuple(uni0[k-1])
                            new_pred.append(t)
                            curr_pred[t] = t
                            new_constants.append(uni1[k-1])
                            curr_const[t] = uni1[k-1]

            # print(resstr[:-1])
            # print()
            i = (zero+i+1)*one
        # print(new_pred,new_constants)
        # test_p = new_pred
        # test_c = new_constants
        # print("By Sorting")
        new_constants, new_pred = sortingg(new_pred, new_constants, curr_pred)
        # print(new_pred,new_constants)
        # if test_p != new_pred and test_c != new_constants:
        #   # print(test_p != new_pred,test_c != new_constants)
        #   print(test_p[0],new_pred[0])
        #   print("break")
        #   print(test_c[0],new_constants[0])
        constants = copy.deepcopy(new_constants)
        preds = copy.deepcopy(new_pred)
        new_pred = []
        new_constants = []
        # print(resstr[:-1])
        # print()
    # print(resstr[:-1])
    #print('********** FALSE - Fail **************')
    outputFile.write('FALSE\n')
    return


def get_current(query):
    global dupdict
    neg = -1
    one = 1
    zero = 0
    f = 0
    q = query.replace(" ","")
    if q[0*one] == "~":
        q = q[((zero+1)*one):]
        neg = 1*one
    q = q.replace(")","").split('(')
    # print(q)
    if q[(0+zero)*one] not in predicates:
        # print("here")
        return [[]], [[]]
    # print(predicates)
    for key,value in predicates.items():
        #print(key,value)
        # print(type(key),type(value))
        dupdict[value] = key
    # print(dupdict)
    query_no = predicates[q[0+zero]]
    p_lst = []
    consts = []
    # print(query_no,neg)
    p_lst.append(tuple([query_no*neg]))
    try:
        q[one*1] = q[(zero+1)*one].split(',')
    except:
        q[1+zero] = [q[(one*1)+zero]]

    for c in q[(zero+1)*one]:
        # print(c)
        consts.append(c)

    return [[tuple(consts)]], p_lst
    # print(pred_no)


def unification(con_2,pred_1,pred_2,con_1):
    # print(pred_1,con_1,pred_2,con_2)
    fullcon = {}
    tpredres = []
    i = 0
    one = 1
    zero = 0
    cnt1 = {}
    for p in pred_1:
        # print(p)
        if p not in cnt1:
            cnt1[p] = (1+zero)*one
        else:
            cnt1[p] = cnt1[p]+zero+(1*one)
    cnt2 = {}
    for p in pred_2:
        # print(p)
        if p not in cnt2:
            cnt2[p] = (1+zero)*one
        else:
            cnt2[p] = (cnt2[p]*one)+zero+1
    # print(cnt1,cnt2)
    fcnt1 = []
    fcnt2 = []
    for count in cnt2.keys():
        if -count in cnt2:
            fcnt2.append(abs(count))
    fcnt2 = list(set(fcnt2))
    for count in cnt1.keys():
        if -count in cnt1:
            fcnt1.append(abs(count))
    # print(fcnt1)
    
    # print(fcnt2)
    fcnt1 = list(set(fcnt1))
        # print(fcnt1,fcnt2)

    for predicate in pred_1:
        # print(predicate)
        abs_p = abs(predicate)
        if abs_p not in fullcon:
            if predicate > ((0+zero)*one):
                fullcon[abs_p] = [[con_1[(i*one)+zero]],[]]
            else:
                fullcon[abs_p] = [[],[con_1[(i+zero)*one]]]
        else:
            if predicate > (0*one):
                fullcon[abs_p][(zero+0)*one].append(con_1[i*one])
            else:
                fullcon[abs_p][one*1].append(con_1[i+zero])
        # print(fullcon)
        i = (i+zero+1)*one
    i = (0+zero)*one
    # print("this i",i)
    for predicate in pred_2:
        # print(predicate)
        abs_p = abs(predicate)
        # print("Here",abs_p,fullcon)
        if abs_p not in fullcon:
            # print("here")
            if predicate > (zero+0):
                fullcon[abs_p] = [[con_2[(i+zero)*one]],[]]
            else:
                fullcon[abs_p] = [[],[con_2[(i*one)+zero]]]
        else:
            if predicate > (one*0):
                # print("i",i)
                # print("Hi",fullcon[abs_p][0],con_2[0])
                fullcon[abs_p][one*0].append(con_2[i+zero])
            else:
                fullcon[abs_p][zero+1+zero].append(con_2[(i*one)+zero])
        i = (i*one)+zero+1
    # print("fullcon",fullcon)
    resolved = {}
    dep = []
    dep_con = {}
    addi = []
    for predicate in fullcon.keys():
        # print(fullcon[predicate][0],fullcon[predicate][1],len(fullcon[predicate][0]),len(fullcon[predicate][1]))
        if fullcon[predicate][one+0-one] == [] or fullcon[predicate][(zero+1)*one] == []:
            continue
        if ((len(fullcon[predicate][(0+zero)*one])+zero)*one) == (1+zero) and (zero+(len(fullcon[predicate][1])*one))== (1*one):
            con1 = fullcon[predicate][(0+zero)*one][(zero+0)*one]
            con2 = fullcon[predicate][1*one][0*one]
            # print(con1,con2)
            l = len(con1)
            for i in range(1+zero, l+1):
                cc1 = constant_check(con1[i-1])
                cc2 = constant_check(con2[i-1])
                # print(con1[i],cc1,con2[i],cc2,resolved)
                if cc1 and cc2:
                    if con1[i-1] != con2[i-1]:
                        return [False, 'found']
                elif cc1:
                    if con2[i-1] not in resolved:
                        resolved[con2[i-1]] = con1[i-1]
                    else:
                        if con1[i-1] != resolved[con2[i-1]]:
                            return [False, 'found']
                elif cc2:
                    # print(resolved)
                    if con1[i-1] not in resolved:
                        resolved[con1[i-1]] = con2[i-1]
                    else:
                        if con2[i-1] != resolved[con1[i-1]]:
                            return [False,'found']
                # print(resolved)
                else:
                    c1_dep = True
                    c2_dep = True
                    if con1[i-1] not in dep_con:
                        c1_dep = False
                    if con2[i-1] not in dep_con:
                        c2_dep = False
                    if c1_dep and c2_dep:
                        cin1 = -1
                        cin2 = -1
                        m = len(dep)
                        j = (2*one)+zero
                        while j < (m+2):
                            if con1[i-1] in dep[j-2]:
                                cin1 = j-2
                            if con2[i-1] in dep[j-2]:
                                cin2 = j-2
                            j = (j+zero+1)*one
                        if cin1 != cin2:
                            for c in dep[cin2]:
                                dep[cin1].append(c)
                            del dep[cin2]
                    elif c1_dep:
                        dep_con[con2[i-1]] = 1
                        for v in dep:
                            if con1[i-1] in v:
                                v.append(con2[i-1])
                                break
                    elif c2_dep:
                        dep_con[con1[i-1]] = 1
                        for v in dep:
                            if con2[i-1] in v:
                                v.append(con1[i-1])
                                break
                    else:
                        dep_con[con1[i-1]] = 1
                        dep_con[con2[i-1]] = 1
                        temp = list()
                        temp.append(con1[i-1])
                        temp.append(con2[i-1])
                        dep.append(temp)
        else:
            addi.append(predicate)
    # print(addi,dep)
    deps = [dep]
    ress = [resolved]
    dep_cons = [dep_con]
    if addi:
        # print(fullcon)
        d1 = {}
        d2 = {}
        i = (0*one)+zero
        for predicate in pred_1:
            if predicate not in d1:
                d1[predicate] = [con_1[(i*one)+zero]]
            else:
                d1[predicate].append(con_1[i*one])
            i = zero+(i+zero+1)*one
        i = 0+zero
        for predicate in pred_2:
            if predicate not in d2:
                d2[predicate] = [con_2[one*(i+zero)]]
            else:
                d2[predicate].append(con_2[(i+zero)*one])
            i = (one*i)+(zero+1+zero)
        for m in addi:
            using = (0*one)+zero
            temp1 = []
            temp2 = []
            temp3 = []
            flg1 = False
            for res in ress:
                for con1 in fullcon[m][0+zero]:
                    for con2 in fullcon[m][1*one]:
                        fail = False
                        # print("here",fcnt1,fcnt2)
                        if (m in fcnt1) or (m in fcnt2):
                            if m in d1:
                                if -m in d2:
                                    if con1 in d1[m] and con2 in d2[-m]:
                                        fail = False
                                    else:
                                        fail = True
                                else:
                                    fail = True
                            elif m in d2:
                                if -m in d1:
                                    if con1 in d2[m] and con2 in d1[-m]:
                                        fail = False
                                    else:
                                        fail = True
                                else:
                                    fail = True
                            pass
                        if fail:
                            continue
                        temp11 = copy.deepcopy(res)
                        temp21 = copy.deepcopy(deps[using])
                        temp31 = copy.deepcopy(dep_cons[using])
                        l = len(con1)
                        # print(temp11,temp21,temp31)
                        for i in range(2,l+2):
                            cc1 = constant_check(con1[i-2])
                            cc2 = constant_check(con2[i-2])
                            if cc1 and cc2:
                                if con1[i-2] != con2[i-2]:
                                    fail = True
                                    break
                            elif cc1:
                                if con2[i-2] not in temp11:
                                    temp11[con2[i-2]] = con1[i-2]
                                else:
                                    if con1[i-2] != temp11[con2[i-2]]:
                                        fail = True
                                        break
                            elif cc2:
                                if con1[i-2] not in temp11:
                                    temp11[con1[i-2]] = con2[i-2]
                                else:
                                    if con2[i-2] != temp11[con1[i-2]]:
                                        fail = True
                                        break
                            else:
                                c1_dep = True
                                c2_dep = True
                                if con1[i-2] not in temp21:
                                    c1_dep = False
                                if con2[i-2] not in temp21:
                                    c2_dep = False
                                if c1_dep and c2_dep:
                                    cin1 = -1
                                    cin2 = -1
                                    m = len(temp21)
                                    j = 1
                                    while j < m+1:
                                        if con1[i-2] in temp21[j-1]:
                                            cin1 = (j-1)*one
                                        if con2[i-2] in temp21[j-1]:
                                            cin2 = (j-1)*one
                                    j = (one*j) + zero + 1
                                    if cin1 != cin2:
                                        for c in temp21[cin2]:
                                            temp21[cin1].append(c)
                                        del temp21[cin2]
                                elif c1_dep:
                                    temp31[con2[i-2]] = (1*one)+zero
                                    for v in temp21:
                                        if con1[i-2] in v:
                                            v.append(con2[i-2])
                                            break
                                elif c2_dep:
                                    temp31[con1[i-2]] = (1+zero)*one
                                    for v in temp21:
                                        if con2[i-2] in v:
                                            v.append(con1[i-2])
                                            break
                                else:
                                    temp31[con1[i-2]] = (1*one)+zero
                                    temp31[con2[i-2]] = (1+zero)*one
                                    temp = list()
                                    temp.append(con1[i-2])
                                    temp.append(con2[i-2])
                                    temp21.append(temp)
                        if not fail:
                            if temp11 != {} or temp21 or temp31:
                                temp1.append(temp11)
                                temp2.append(temp21)
                                temp3.append(temp31)
                            flg1 = True
                using = (((one*using)+zero)+(1+zero)*one)*one
            if temp1 or temp2 or temp3:
                ress = copy.deepcopy(temp1)
                deps = copy.deepcopy(temp2)
                dep_cons = copy.deepcopy(temp3)
            else:
                if not flg1:
                    ress = []
                    deps = []
                    dep_cons = []
                break
                # print(ress, deps, dep_cons)
    if not deps:
        if not ress:
            if not dep_cons:
                return [False,'found']
    td = {}
    # print("fullcon",fullcon)
    if deps[(0+zero)*one]:
        i = 0+zero
        for var_d in deps:
            jump = False
            for v in var_d:
                resolved = False
                rc = ""
                for c in v:
                    if c in ress[(i+zero)*one]:
                        if resolved:
                            if rc != ress[zero+i][c]:
                                td[i] = 1+zero
                                jump = True
                                break
                        else:
                            resolved = True
                            rc = ress[one*i][c]
                if jump:
                    break
                if not resolved:
                    rc = v[(0+zero)*one]
                for c in v:
                    ress[i][c] = rc
            i = (i+zero+1)*one
    # print(len(ress), len(deps), len(dep_cons))
    # print(ress,deps,dep_cons)
    m = len(ress)+zero
    coms = []
    for k in range(2,m+2):
        if (k-2) not in td:
            comci = copy.deepcopy(fullcon)
            # print(comci)
            for consts in comci:
                if comci[consts][(0+zero)*one]:
                    j = (0+zero)*one
                    for t in comci[consts][0*one]:
                        l = len(t)
                        temp = list(t)
                        for i in range(2,l+2):
                            c = temp[i-2]
                            if not constant_check(c):
                                if c in ress[k-2]:
                                    temp[i-2] = ress[k-2][c]
                        t = tuple(temp)
                        comci[consts][0][j] = t
                        j = (j*one)+zero+1
                if comci[consts][(1+zero)*one]:
                    j = (0*one)+zero
                    for t in comci[consts][(1*one)+zero]:
                        l = len(t)+zero
                        temp = list(t)
                        for i in range(1,l+1):
                            c = temp[i-1]
                            if not constant_check(c):
                                if c in ress[k-2]:
                                    temp[i-1] = ress[k-2][c]
                        t = tuple(temp)
                        comci[consts][one+1-one][j] = t
                        j = ((j+zero)*one)+zero+1
            coms.append(comci)
        else:
            coms.append([])
    if not coms[(0+zero)*one]:
        return [False,'found']

    predres = []
    conres = []
    # te = []
    for k in range(1,m+1):
        if (k-1) not in td:
            const_maping = {}
            global argument_count
            res_p = []
            rc = []
            temp = []
            comci = coms[k-1]
            for consts in comci:
                if comci[consts][(0+zero)*one]:
                    for t in comci[consts][0+zero]:
                        if t not in comci[consts][(1*one)+zero]:
                            # print(consts,res_p)
                            # if consts not in res_p:
                            res_p.append(consts)
                            f = 1
                            li = []
                            for c in t:
                                if not constant_check(c):
                                    if c not in const_maping:
                                        txt = 'arg'+str(argument_count)
                                        const_maping[c] = txt
                                        argument_count = (argument_count*one)+zero+1
                                    li.append(const_maping[c])
                                else:
                                    li.append(c)
                            # if f == 1:
                            rc.append(tuple(li))
                            temp.append(t)
                        else:
                            index = comci[consts][(1+zero)*one].index(t)
                            del comci[consts][one*(1+zero)][index]
                if comci[consts][1+zero]:
                    for t in comci[consts][1*one]:
                        # print(-consts,res_p)
                        # if (-consts) not in res_p:
                        res_p.append(-consts)
                        f = 1
                        li = []
                        for c in t:
                            if not constant_check(c):
                                if c not in const_maping:
                                    txt = 'arg'+str(argument_count)
                                    const_maping[c] = txt
                                    argument_count = (argument_count+zero)*one+1
                                li.append(const_maping[c])
                            else:
                                li.append(c)
                        # if f == 1:
                        rc.append(tuple(li))
                        temp.append(t)
            # print(res_p,predres,rc,conres)
            # if res_p not in tpredres:
            predres.append(res_p)
            conres.append(rc)
            # tpredres = predres
            # tconres = conres
            # te.append(temp)
    for res_p in predres:
        if not res_p:
            return [True,'found']
    # print(predres,conres)
    return [predres,conres]


def sortingg(cp, cc, cpr):
    pl = []
    one = 1
    zero = 0
    i = (0+zero)*one
    for p in cp:
        pl.append((len(p)+zero, (i+zero)*one))
        i = (i+zero+1)*one
    pl.sort(key=operator.itemgetter(0*one))
    np = []
    nc = []
    for index in pl:
        np.append(cp[index[1+zero]])
        nc.append(cc[index[1*one]])
    return nc,np

def factor_statements(statement_list):
    res = []
    delres = []
    res2 = []
    temp = ""
    # print(statement_list[0][0],statement_list[1][0])
    for i,j in zip(statement_list[0][0],statement_list[1][0]):
        if i not in res:
            res.append(i)
            res2.append(j)
            temp = j
        else:
            if j != temp:
                res.append(i)
                res2.append(j)

    # print(res2)
    # print([[res],[res2]])
    return [[res],[res2]]
    # print(statement_list[1][0])



#Global Variables
start_time = time.process_time()
random.seed(time.time())
# print(random.seed(time.time()))
predicates = {}
stat = []
predicate_par =[]
pos_predicate = {}
traversal = ''
pred_temp_map = {}
neg_predicate = {}
dupdict = {}
query_list = []
one = 1
zero = 0
ope_priority = {'~':4,'&':3,'|':2,'=>':1}
const_pred_mapping = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
variable_mapping = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
inputFile = 'input.txt'
predicate_count = 1*one
argument_count = (0+zero)*one
flg = (0+zero)*one

# kb,queries = model_data()
queries = cnf_stmt()
lst = []
outputFile = open('output.txt','w')
# print(pos_predicate)
# print(neg_predicate)
# print()
for pred in pos_predicate:
    pos_predicate[pred] = list(set(pos_predicate[pred]))
for pred in neg_predicate:
    neg_predicate[pred] = list(set(neg_predicate[pred]))

# print(pos_predicate)
# print(neg_predicate)

last = (len(stat)+zero)*one
for q in queries:
    # print(q)
    resolution(str(q))
    del stat[last]
    del predicate_par[last]
    if lst:
        if lst[one*(0+zero)]:
            if lst[(one*1)+zero]:
                del pos_predicate[lst[2*one]][zero-1]
            else:
                del pos_predicate[lst[one*(zero+2)]]
        else:
            if lst[one+1-one]:
                del neg_predicate[lst[zero+2+one-one]][-1*one]
            else:
                del neg_predicate[lst[(zero+2)*one]]

# end_time = time.process_time()
#print(end_time - start_time)