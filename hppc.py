import random
import os
import shutil
import math
import warnings
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def predictor(training_trace):
    data = pd.Series(training_trace)
    model = SimpleExpSmoothing(data)
    model_fit = model.fit(smoothing_level=0.2, optimized=False)
    return model_fit.forecast(1).values[0]

def functionPriority(i):
    su_ratio = (speedup_list[i] - min(speedup_list)) / (max(speedup_list) - min(speedup_list))
    mem_ratio = (mem_list[i] - min(mem_list)) / (max(mem_list) - min(mem_list))
    try:
        time_ratio = (tit_list[i] - min(tit_list)) / (max(tit_list) - min(tit_list))
    except:
        time_ratio = 0
    priority = ((1 - su_ratio) + (1 - mem_ratio) + time_ratio) / 3
    return priority

def montecarlo(run_time, number):
    if number == 0:
        return 0
    n = max_round
    parallel_count = []
    for i in range(n):
        arr = []
        for j in range(number):
            random_time = random.uniform(0, 60 - run_time)
            arr.append(random_time)
        sorted_arr = sorted(arr)
        max_parallel = 1
        c_list = [0 for c in range(number)]
        for j in range(len(sorted_arr)):
            if j == 0:
                c_list[0] = run_time
            else:
                parallel = 0
                flag = 0
                interval_time = sorted_arr[j] - sorted_arr[j-1]
                for ii in range(len(c_list)):
                    if c_list[ii] == 0:
                        if flag == 0:
                            c_list[ii] = run_time
                            parallel += 1
                            flag = 1
                    else:
                        if c_list[ii] <= interval_time:
                            c_list[ii] = 0
                            if flag == 0:
                                c_list[ii] = run_time
                                parallel += 1
                                flag = 1
                        else:
                            c_list[ii] = c_list[ii] - interval_time
                            parallel += 1
                if parallel > max_parallel:
                    max_parallel = parallel
        parallel_count.append(max_parallel)
    sorted_parallel = sorted(parallel_count)
    container = sorted_parallel[math.ceil(n * alpha) - 1]
    return container

def pre_warmer(t):
    global high_memory
    global low_memory
    global high_servers
    global low_servers
    score_list = []
    for i in range(len(trace_list)):
        score = functionPriority(i)
        score_list.append(score)
    sorted_score_list = sorted(score_list, reverse=True)
    if t > 0:
        w_high_end_list.append(w_high_end_list[t - 1] + w_high_starting_list[t - 1])
        if w_high_starting_list[t - 1] == -1:
            del high_memory[-1]
            del high_servers[-1]
        elif w_high_starting_list[t - 1] == 1:
            high_memory.append(mem_server)
            high_servers.append([])
        w_low_end_list.append(w_low_end_list[t - 1] + w_low_starting_list[t - 1])
        if w_low_starting_list[t - 1] == -1:
            del low_memory[-1]
            del low_servers[-1]
        elif w_low_starting_list[t - 1] == 1:
            low_memory.append(mem_server)
            low_servers.append([])
        if w_high_starting_list[t - 1] == -1 or w_low_starting_list[t - 1] == -1:
            w_server_closed[0] = 0
        if w_high_starting_list[t - 1] == 1 or w_low_starting_list[t - 1] == 1:
            w_server_opened[0] = 0
    else:
        pass
    high_server = [mem_server for h in range(w_high_end_list[t] - 1)]
    low_server = [mem_server for l in range(w_low_end_list[t])]
    high_server.append(mem_server - cold_start_memory)
    zero_index = []
    judge = 0
    for score in sorted_score_list:
        high_server.sort()
        low_server.sort()
        i = score_list.index(score)
        pred_value = predicted_list[i][t]
        if pred_value == 0:
            w_high_container_list[i].append(0)
            w_low_container_list[i].append(0)
            zero_index.append(i)
        else:
            c_high = montecarlo(exe_time_costly[i], pred_value)
            con_high = c_high
            h = 0
            while c_high > 0:
                if h >= len(high_server):
                    break
                con = math.floor(high_server[h] / mem_list[i])
                con = min(con, c_high)
                c_high -= con
                high_server[h] -= (con * mem_list[i])
                h += 1
            if c_high == 0:
                w_high_container_list[i].append(con_high)
                w_low_container_list[i].append(0)
            else:
                w_high_container_list[i].append(con_high - c_high)
                c_low = montecarlo(exe_time_cheap[i], pred_value)
                c_rep = round(c_high * (c_low / con_high))
                con_low = c_rep
                l = 0
                while c_rep > 0:
                    if l >= len(low_server):
                        break
                    con = math.floor(low_server[l] / mem_list[i])
                    con = min(con, c_rep)
                    c_rep -= con
                    low_server[l] -= (con * mem_list[i])
                    l += 1
                w_low_container_list[i].append(con_low - c_rep)
                if c_rep > 0:
                    judge = 1
    if mem_server in high_server or mem_server in low_server:
        w_server_closed[0] += 1
        w_server_opened[0] = 0
        if w_server_closed[0] >= 5:
            if len(low_server) < len(high_server):
                w_low_starting_list.append(0)
                w_high_starting_list.append(-1)
            else:
                w_low_starting_list.append(-1)
                w_high_starting_list.append(0)
        else:
            w_low_starting_list.append(0)
            w_high_starting_list.append(0)
    else:
        w_server_closed[0] = 0
        if judge == 0:
            w_low_starting_list.append(0)
            w_high_starting_list.append(0)
        elif judge == 1:
            w_server_opened[0] += 1
            if w_server_opened[0] >= 3:
                if len(low_server) < len(high_server):
                    w_low_starting_list.append(1)
                    w_high_starting_list.append(0)
                else:
                    w_low_starting_list.append(0)
                    w_high_starting_list.append(1)
            else:
                w_low_starting_list.append(0)
                w_high_starting_list.append(0)
    high_con_had = [0 for i in range(len(trace_list))]
    low_con_had = [0 for i in range(len(trace_list))]
    for s in range(len(high_servers)):
        for c in high_servers[s]:
            high_con_had[c] += 1
    for s in range(len(low_servers)):
        for c in low_servers[s]:
            low_con_had[c] += 1
    for i in range(len(trace_list)):
        while high_con_had[i] > w_high_container_list[i][t]:
            for m_ind in range(len(high_memory)):
                high_memory[m_ind] += m_ind * 0.0001
            sorted_mem = sorted(high_memory, reverse=True)
            flag = 0
            for m in sorted_mem:
                ind = high_memory.index(m)
                for c in range(len(high_servers[ind])):
                    if high_servers[ind][c] == i:
                        del high_servers[ind][c]
                        high_memory[ind] += mem_list[i]
                        flag = 1
                        high_con_had[i] -= 1
                        break
                if flag == 1:
                    break
            for jj in range(len(high_memory)):
                high_memory[jj] = int(high_memory[jj])
        while low_con_had[i] > w_low_container_list[i][t]:
            for m_ind in range(len(low_memory)):
                low_memory[m_ind] += m_ind * 0.0001
            sorted_mem = sorted(low_memory, reverse=True)
            flag = 0
            for m in sorted_mem:
                ind = low_memory.index(m)
                for c in range(len(low_servers[ind])):
                    if low_servers[ind][c] == i:
                        del low_servers[ind][c]
                        low_memory[ind] += mem_list[i]
                        flag = 1
                        low_con_had[i] -= 1
                        break
                if flag == 1:
                    break
            for jj in range(len(low_memory)):
                low_memory[jj] = int(low_memory[jj])
    for i in range(len(trace_list)):
        if high_con_had[i] < w_high_container_list[i][t]:
            high_con_added[i] = w_high_container_list[i][t] - high_con_had[i]
        if low_con_had[i] < w_low_container_list[i][t]:
            low_con_added[i] = w_low_container_list[i][t] - low_con_had[i]
    high_memory_copy = high_memory.copy()
    high_servers_copy = high_servers.copy()
    low_memory_copy = low_memory.copy()
    low_servers_copy = low_servers.copy()
    judgeee = 0
    for m_ind in range(len(high_memory_copy)):
        high_memory_copy[m_ind] += m_ind * 0.0001
    for m_ind in range(len(low_memory_copy)):
        low_memory_copy[m_ind] += m_ind * 0.0001
    for m in mem_sorted_list:
        ind = mem_list.index(m)
        high_con_add = high_con_added[ind]
        while high_con_add > 0:
            flag = 0
            sorted_mem = sorted(high_memory_copy)
            for mm in sorted_mem:
                if mm >= m:
                    iii = high_memory_copy.index(mm)
                    high_memory_copy[iii] -= m
                    high_servers_copy[iii].append(ind)
                    flag = 1
                    break
            high_con_add -= 1
            if flag == 0:
                judgeee = 1
        low_con_add = low_con_added[ind]
        while low_con_add > 0:
            flag2 = 0
            sorted_mem = sorted(low_memory_copy)
            for mm in sorted_mem:
                if mm >= m:
                    iii = low_memory_copy.index(mm)
                    low_memory_copy[iii] -= m
                    low_servers_copy[iii].append(ind)
                    flag2 = 1
                    break
            low_con_add -= 1
            if flag2 == 0:
                judgeee = 1
    for jj in range(len(high_memory_copy)):
        high_memory_copy[jj] = int(high_memory_copy[jj])
    for jj in range(len(low_memory_copy)):
        low_memory_copy[jj] = int(low_memory_copy[jj])
    if judgeee == 0:
        high_memory = high_memory_copy.copy()
        high_servers = high_servers_copy.copy()
        low_memory = low_memory_copy.copy()
        low_servers = low_servers_copy.copy()
    else:
        high_memory = [mem_server for h in range(w_high_end_list[t] - 1)]
        low_memory = [mem_server for l in range(w_low_end_list[t])]
        high_memory.append(mem_server - cold_start_memory)
        high_memory.sort()
        high_servers = [[] for h in range(w_high_end_list[t])]
        low_servers = [[] for h in range(w_low_end_list[t])]
        for score in sorted_score_list:
            i = score_list.index(score)
            high_con_added[i] = w_high_container_list[i][t]
            low_con_added[i] = w_low_container_list[i][t]
            high_con_add = high_con_added[i]
            low_con_add = low_con_added[i]
            while high_con_add > 0:
                sorted_mem = sorted(high_memory)
                m = mem_list[i]
                for mm in sorted_mem:
                    if mm >= m:
                        iii = high_memory.index(mm)
                        high_memory[iii] -= m
                        high_servers[iii].append(i)
                        break
                high_con_add -= 1
            while low_con_add > 0:
                sorted_mem = sorted(low_memory)
                m = mem_list[i]
                for mm in sorted_mem:
                    if mm >= m:
                        iii = low_memory.index(mm)
                        low_memory[iii] -= m
                        low_servers[iii].append(i)
                        break
                low_con_add -= 1
    high_server = sorted(high_memory)
    low_server = sorted(low_memory)
    zero_index.reverse()
    for z in range(len(zero_index) - 1, -1, -1):
        flag = 0
        m = mem_list[zero_index[z]]
        for h in range(len(high_server)):
            if high_server[h] >= m:
                high_server[h] -= m
                w_high_container_list[zero_index[z]][t] += 1
                high_con_added[zero_index[z]] += 1
                flag = 1
                break
        if flag == 0:
            for h in range(len(low_server)):
                if low_server[h] >= m:
                    low_server[h] -= m
                    w_low_container_list[zero_index[z]][t] += 1
                    low_con_added[zero_index[z]] += 1
                    break
    for z in range(len(zero_index) - 1, -1, -1):
        flag = 0
        m = mem_list[zero_index[z]]
        for h in range(len(high_server)):
            if high_server[h] >= m:
                high_server[h] -= m
                w_high_container_list[zero_index[z]][t] += 1
                high_con_added[zero_index[z]] += 1
                flag = 1
                break
        if flag == 0:
            for h in range(len(low_server)):
                if low_server[h] >= m:
                    low_server[h] -= m
                    w_low_container_list[zero_index[z]][t] += 1
                    low_con_added[zero_index[z]] += 1
                    break
    for s in sorted_score_list:
        ini = score_list.index(s)
        m = mem_list[ini]
        h = 0
        while h < len(high_server):
            if high_server[h] >= m:
                high_server[h] -= m
                w_high_container_list[ini][t] += 1
                high_con_added[ini] += 1
            else:
                h += 1
    for s in sorted_score_list:
        ini = score_list.index(s)
        m = mem_list[ini]
        h = 0
        while h < len(low_server):
            if low_server[h] >= m:
                low_server[h] -= m
                w_low_container_list[ini][t] += 1
                low_con_added[ini] += 1
            else:
                h += 1

def scheduler(sorted_arr, t):
    high_container_list = [[] for i in range(len(trace_list))]
    low_container_list = [[] for i in range(len(trace_list))]
    for i in range(len(trace_list)):
        w_service_time_list[i].append(0)
        high_c = w_high_container_list[i][t]
        low_c = w_low_container_list[i][t]
        high_container_list[i].append(10000)
        low_container_list[i].append(10000)
        high_new = high_con_added[i]
        low_new = low_con_added[i]
        for c in range(high_c):
            if high_new > 0:
                high_container_list[i].append(time_used + cs_time_costly[i])
                high_new -= 1
            else:
                high_container_list[i].append(0)
        for c in range(low_c):
            if low_new > 0:
                low_container_list[i].append(time_used + cs_time_cheap[i])
                low_new -= 1
            else:
                low_container_list[i].append(0)
    first_arr = [sorted_arr[i][0] for i in range(len(trace_list))]
    for i in range(len(trace_list)):
        sorted_arr[i].pop(0)
    reserved_memory = cold_start_memory
    try:
        for c in range(len(reserved_space)):
            reserved_memory -= mem_list[reserved_space[c]]
            reserved_containers[c] = 0
    except:
        pass
    last_time = 0
    first_time = min(first_arr)
    del_con = []
    while first_time < 100:
        interval_time = first_time - last_time
        try:
            for c in range(len(reserved_containers)):
                reserved_containers[c] = max(0, reserved_containers[c] - interval_time)
        except:
            pass
        for i in range(len(trace_list)):
            for h in range(len(high_container_list[i])):
                high_container_list[i][h] = max(0, high_container_list[i][h] - interval_time)
            for l in range(len(low_container_list[i])):
                low_container_list[i][l] = max(0, low_container_list[i][l] - interval_time)
        ind = first_arr.index(first_time)  # Function index
        min_cs_time = 10000
        del_con.clear()
        min_cs_con = 0
        m = reserved_memory
        min_cs_way = 0
        for c in range(len(reserved_space)):
            if reserved_containers[c] == 0 and reserved_space[c] != ind:
                m += mem_list[reserved_space[c]]
            if reserved_space[c] == ind:
                if (reserved_containers[c] + exe_time_costly[ind]) <= min_cs_time:
                    min_cs_time = reserved_containers[c] + exe_time_costly[ind]
                    min_cs_con = c
                    min_cs_way = 1
        if m >= mem_list[ind]:
            if (cs_time_costly[ind] + exe_time_costly[ind]) < min_cs_time:
                min_cs_way = 2
                min_cs_time = cs_time_costly[ind] + exe_time_costly[ind]
                mm = reserved_memory
                if mm < mem_list[ind]:
                    for c in range(len(reserved_space)):
                        if reserved_containers[c] == 0 and reserved_space[c] != ind:
                            del_con.append(c)
                            mm += mem_list[reserved_space[c]]
                            if mm >= mem_list[ind]:
                                break
        else:
            judge = 0
            cs_indices = [i for i in range(len(reserved_containers))]
            cs_containers = reserved_containers.copy()
            combined = list(zip(cs_containers, cs_indices))
            sorted_combined = sorted(combined, key=lambda x: x[0])
            sorted_list1, sorted_list2 = zip(*sorted_combined)
            sorted_list1 = list(sorted_list1)
            sorted_list2 = list(sorted_list2)
            for cc in range(len(sorted_list1)):
                if sorted_list1[cc] == 0 or reserved_space[sorted_list2[cc]] == ind:
                    continue
                c = sorted_list2[cc]
                wait_time = sorted_list1[cc]
                m += mem_list[reserved_space[c]]
                del_con.append(c)
                if m >= mem_list[ind]:
                    judge = 1
                    m = reserved_memory
                    for cccc in del_con:
                        m += mem_list[cccc]
                    if m >= mem_list[ind]:
                        if wait_time + cs_time_costly[ind] + exe_time_costly[ind] < min_cs_time:
                            min_cs_way = 3
                            min_cs_time = wait_time + cs_time_costly[ind] + exe_time_costly[ind]
                    for ccc in range(len(reserved_containers)):
                        if reserved_containers[ccc] == 0:
                            del_con.append(ccc)
                            m += mem_list[reserved_space[ccc]]
                            if m >= mem_list[ind]:
                                if wait_time + cs_time_costly[ind] + exe_time_costly[ind] < min_cs_time:
                                    min_cs_way = 3
                                    min_cs_time = wait_time + cs_time_costly[ind] + exe_time_costly[ind]
                                break
                if judge == 1:
                    break
        if min_cs_way == 1:
            del_con.clear()
            del_con.append(min_cs_con)
        shortest_time = [min(high_container_list[ind]) + exe_time_costly[ind], min(low_container_list[ind]) + exe_time_cheap[ind],
                         min_cs_time]
        min_time = min(shortest_time)
        way = shortest_time.index(min_time)
        if min_time + first_time > 60 + 0.7 * exe_time_costly[ind]:
            way = 2
        if way == 0:
            w_service_time_list[ind][t] += min_time
            high_container_list[ind][high_container_list[ind].index(min(high_container_list[ind]))] += exe_time_costly[ind]
        elif way == 1:
            w_service_time_list[ind][t] += min_time
            low_container_list[ind][low_container_list[ind].index(min(low_container_list[ind]))] += exe_time_cheap[ind]
        else:
            w_service_time_list[ind][t] += min_cs_time
            del_con.sort(reverse=True)
            if len(del_con) > 0:
                for c in del_con:
                    del reserved_containers[c]
                    reserved_memory += mem_list[reserved_space[c]]
                    del reserved_space[c]
            reserved_space.append(ind)
            reserved_memory -= mem_list[ind]
            reserved_containers.append(min_cs_time)
        last_time = first_time
        first_arr[ind] = sorted_arr[ind][0]
        sorted_arr[ind].pop(0)
        first_time = min(first_arr)

def write_results(dir):
    methoddir = dir
    if os.path.exists(methoddir):
        shutil.rmtree(methoddir)
    os.makedirs(methoddir)
    with open(methoddir + '/keep alive cost.txt', 'w') as f:
        for item in w_keepalive_cost_list:
            f.write("%s\n" % item)
    for i in range(len(trace_list)):
        funcdir = methoddir + "/Function" +str(i + 1)
        with open(funcdir + ' service time.txt', 'w') as f:
            for item in w_service_time_list[i]:
                f.write("%s\n" % item)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")    ##
    # Prepared by IceBreaker
    function_index_list = [(i + 1) for i in range(12)]
    filename_list = []
    for item in function_index_list:
        filename_list.append("./traces/f" + str(item) + ".txt")
    trace_list = []
    for file in filename_list:
        with open(file) as f1:
            trace = f1.read().splitlines()
            trace = [float(i) for i in trace]
        trace_list.append(trace)
    trace_list = [[i * 2 for i in trace] for trace in trace_list] #
    exe_time_costly = [3.03, 0.72, 2.985, 0.7, 3.04, 0.71, 2.99, 0.71, 3.02, 0.72, 3.03, 0.69]
    exe_time_cheap = [3.4490861279670932, 0.8333479646247067, 3.4859716333219093, 0.7878767087236034, 3.546308500142927,
                      0.8180562647034706, 3.395032713743947, 0.8175997881773507, 3.4235695900618275, 0.8094754467444711,
                      3.423391272136164, 0.799949318489927]
    cs_time_costly = [2.3138078812772824, 0.5351947442960704, 2.328516853298586, 0.5336823548924382, 2.305156829309391,
                      0.5536490620847443, 2.293040393192761, 0.5567880110232695, 2.4029879943159442, 0.5292653936701095,
                      2.313691032368492, 0.5508219006501083]
    cs_time_cheap = [2.6400779822155878, 0.6322031988231528, 2.428365813380892, 0.5694407471685499, 2.541300418642583,
                     0.5823974096586262, 2.4606131250934133, 0.5977289774037928, 2.5075122606395377, 0.6044489158399579,
                     2.551358071916901, 0.5707059956824149]
    mem_list = [507, 499, 177, 173, 124, 125, 505, 496, 178, 172, 122, 126]
    mem_sorted_list = mem_list.copy()
    mem_sorted_list.sort(reverse=True)
    cost_per_sec_per_mb = [0.01475 / (3600 * 1024), 0.0084 / (3600 * 1024)]
    # Prediction results
    pred_list = []
    for i in range(len(trace_list)):
        file = "./prediction results/SES/function" + str(i) + "/predicted value.txt"
        with open(file) as f1:
            pred = f1.read().splitlines()
            pred = [int(j) for j in pred]
        pred_list.append(pred)
    speedup_list = [(exe_time_cheap[i] / exe_time_costly[i]) for i in range(len(exe_time_costly))]
    mem_server = int(2 * 1024)
    real_list = [[] for i in range(len(trace_list))]
    predicted_list = [[] for i in range(len(trace_list))]
    sliding_window = 60
    alpha = 0.9
    max_round = 50
    time_used = 0.001 + max_round / 100 * 0.0004
    cold_start_memory = int(0.5 * 1024)
    training_trace = [trace_list[i][:sliding_window] for i in range(len(trace_list))]
    w_server_closed = [0]
    w_server_opened = [0]
    reserved_space = []
    reserved_containers = []
    high_memory = [mem_server - cold_start_memory, mem_server]
    low_memory = [mem_server, mem_server]
    high_servers = [[], []]
    low_servers = [[], []]
    w_high_end_list = [2]
    w_low_end_list = [2]
    w_high_starting_list = []
    w_low_starting_list = []
    w_high_container_list = [[] for i in range(len(trace_list))]
    w_low_container_list = [[] for i in range(len(trace_list))]
    w_keepalive_cost_list = []
    w_service_time_list = [[] for i in range(len(trace_list))]
    for t in range(0, len(trace_list[0]) - sliding_window):
        if t % 100 == 0:
            print(t)
        high_con_added = [0 for i in range(len(trace_list))]
        low_con_added = [0 for i in range(len(trace_list))]
        tit_list = []
        for i in range(len(trace_list)):
            real_value = round(trace_list[i][t + sliding_window])
            real_list[i].append(real_value)
            if t > 0:
                training_trace[i].append(real_list[i][t - 1])
                training_trace[i] = training_trace[i][1:]
            else:
                pass
            pred_value = round(pred_list[i][t]) # Load prediction results
            # pred_value = predictor(training_trace[i]) # Predict
            # pred_value = max(0, round(pred_value))
            predicted_list[i].append(pred_value)
            tit_list.append(pred_value * (exe_time_costly[i] + cs_time_costly[i]))
        pre_warmer(t)  # WarmBooth
        sorted_arr = []
        for i in range(len(trace_list)):
            real_value = real_list[i][t]
            arr = [10000]
            for n in range(real_value):
                random_time = random.uniform(0, 60)
                arr.append(random_time)
            sorted_arr.append(sorted(arr))
        scheduler(sorted_arr, t)  ## WarmBooth
        w_keepalive_cost_list.append((w_high_end_list[t] + max(0, w_high_starting_list[t])) * cost_per_sec_per_mb[0] * 60 * mem_server + (w_low_end_list[t] + max(0, w_low_starting_list[t])) * cost_per_sec_per_mb[1] * 60 * mem_server)
    print("Cost: " + str(sum(w_keepalive_cost_list)))
    total_time = 0
    for i in range(len(trace_list)):
        total_time += sum(w_service_time_list[i])
    print("Time: " + str(total_time))
    write_results("./experimental results")