#!/usr/bin/env python
# -*- coding: utf-8 -*-
# get_call_data.py
import datetime
import json
import requests
import time
import os
import socket
import urllib

# from domain_names import get_all_domains


out_format = 'json'  # or txt

dst_dir = 'data/intent_corpus'

REQ_TIMEOUT = 2  # sec


def dump_to_jsonfile(jsondata, filename):
    if jsondata:
        print('writing file', filename)
        with open(filename, 'w', encoding='utf-8') as fout:
            json.dump(jsondata, fout, sort_keys=True,
                      ensure_ascii=False, indent=4)


def dump_to_txtfile(data, filename):
    if data:
        print('writing file', filename)
        with open(filename, 'a', encoding='utf-8') as fout:
            for j in data:
                fout.write('%s\n' % j)


# 对数据格式进行处理
def get_all_content_json(content):
    ret = ''
    if content:
        if isinstance(content, dict):
            data = content.get('data', dict)
            for curJson in data:
                val = curJson.get('content', '').strip().replace('(无声音)', '')
                if val.strip() != '':
                    val = val + '####'
                else:
                    val = ''
                ret += val
            ret = ret[:-4]
    return ret.replace(' ', '')


# 对数据格式进行处理
def get_all_content_txt(content):
    ret = []
    # airet = []
    if content:
        if isinstance(content, dict):
            data = content.get('data', dict)
            for j in data:
                content = j.get('content', '').strip()
                # print(content)
                index = j.get('file', '').split('/')[-1]
                if content and '(无声音)' not in content:
                    val = index+'\t'+j.get('answer', '') + '\t' + content
                    get_wav(j.get('file', ''))
                    ret.append(val)
    print('get_all_content_txt got', len(ret))
    return ret


def request_url(url):
    try:
        res = requests.get(url, timeout=REQ_TIMEOUT)
    except Exception as e:
        print('in request_url', url, str(type(e)), str(e))
        return None
    return res


# 取得对应模板一天的回复数据
def get_voice_content(cfg_name="bjxdhf", add_time="2018-02-22", save_file=False, domain_names=None):
    if domain_names is None:
        domain_names = ['nj', 'nj2', 'huzhou', 'phone']

    # print('domain_names:', domain_names)
    tpl = "http://%s.btows.com/api/ai_get_temp_voice_content.php?temp_str="
    url_list = []
    for j in domain_names:
        url_list.append(tpl % j + cfg_name + "_en&add_time=" + add_time)

    # print('url_list:', url_list)

    content = '' if out_format == 'json' else []
    for url in url_list:
        print('url:', url)
        res = request_url(url)
        if res:
            if res.status_code == 200:
                # print('res.content:', res.content)
                result = json.loads(res.content.decode())
                data_cnt = len(result['data'])
                if data_cnt > 0:
                    # print('result.data:', data_cnt, result)
                    if out_format == 'json':
                        content += get_all_content_json(result)
                    else:
                        content += get_all_content_txt(result)
            else:
                print('bad res.status_code:', res.status_code)
        else:
            print('bad res')

    if save_file:
        if content:
            if out_format == 'json':
                dump_to_jsonfile(content, dst_dir + cfg_name +
                                 '@'+add_time.replace('-', '_') + "_content.json")
                content = content.replace(' ', '')
            else:
                dump_to_txtfile(content, dst_dir + cfg_name + '@' +
                                add_time.replace('-', '_') + "_content.txt")
    return content


# 取得一段时间对应模板的回复数据
def get_voice_content_muti(cfg_name="gjdk", start_time="2010-02-02",
                           end_time="2018-03-08", save_file=False, domain_names=None, save_file_name=None):
    all_content = '' if out_format == 'json' else []
    day_plus = datetime.timedelta(days=1)
    begin_date = datetime.datetime.strptime(start_time, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_time, '%Y-%m-%d')
    cur_date = begin_date
    while cur_date <= end_date:
        time.sleep(1)
        cur_content = get_voice_content(cfg_name, cur_date.strftime(
            '%Y-%m-%d'), domain_names=domain_names)
        print("cfg %s date:" % cfg_name +
              cur_date.strftime('%Y-%m-%d') + ' cnt: ' + str(len(cur_content)))
        cur_date += day_plus
        all_content += cur_content

    if len(all_content) > 0:
        if save_file:
            if out_format == 'json':
                dump_to_jsonfile(all_content, dst_dir +
                                 cfg_name + "_content.json")
            else:
                out_fn = save_file_name if save_file_name else dst_dir + cfg_name + "_content.txt"
                dump_to_txtfile(all_content, out_fn)
        # all_content = all_content.replace(' ', '')
        return all_content
    else:
        print('cfg %s get_voice_content_muti: no data between %s and %s' % (cfg_name,
                                                                            start_time,
                                                                            end_time))


def get_user_sentence_data(cfg_name, domain_names=None, save_file_name=None):
    global out_format
    out_format = 'txt'
    models_empty = []
    # cfg_name = 'xcdk'  # gjdk cxqdk thdk_base_vm mdtjdk xcdk
    # domain_names = None  # default: ['nj', 'nj2', 'huzhou', 'phone']
    if domain_names is None:
        # domain_names = ['nj', 'nj4']
        domain_names = ['nj', 'nj2', 'huzhou', 'phone']
    # start_time = "2017-08-01", end_time = "2017-12-31", save_file = True,
    print('================= getting', cfg_name)
    result = get_voice_content_muti(cfg_name=cfg_name,
                                    # start_time="2018-03-19", end_time="2018-03-19",
                                    # start_time="2017-08-01", end_time="2017-12-31",

                                    start_time="2017-06-01", end_time="2018-06-27",
                                    # start_time="2018-03-12", end_time="2018-03-12",
                                    save_file=True,
                                    domain_names=domain_names,
                                    save_file_name=save_file_name)

    print('result:', len(result) if result else 0)
    # r = len(result) if result else 0
    # if r == 0:
    #     f = open('log.txt','a')
    #     f.write(cfg_name + '\n')
    #     f.close()


def get_user_sentence_data_cfgs():
    # cfg_names = get_all_cfgs()
    # "bayrdk", "bjhxdk", "bjpaphdk", "brzjdk", "bskjdk2", "cddk", "cgdk", "clmdk", "cxqdk", "cyjdk",
    cfg_names = ['fxr']
    # cfg_names = ['fxr', 'bjdxdk', 'hgydk', 'dxydk', 'ylmzpadk', 'txlhzs', 'mddk', 'sdjrdk', 'mmjzy', 'yfy',
    #              'szwhjddk', 'zlmdk', 'jjldk', 'jydk6', 'axdk', 'sjjr', 'njjyzx', 'xmsyxx', 'ksdk', 'zpjfdk', 'xhcf', 'tnjf',
    #              'jshc', 'ztxxcdk', 'ddmdk', 'xndk', 'whxrkdk', 'drdk', 'zwhd', 'xmdk', 'fpxdk2', 'dashengjinrong', 'dkgxb2',
    #              'tnjf2', 'sxydk', 'qbjdk', 'warzdk', 'yxck', 'bszc2', 'jjdktz', 'zyr', 'bjllhd', 'dgjf', 'zcjf', 'jdytdk', 'jzdk', 'gxxedk',
    #              'bkqb', 'hhyjr', 'thjr', 'xyzc', 'ycjrdk', 'xhjfdk', 'bcxxkj', 'zhbwl', 'hzbfx2', 'njhjjr2', 'hljpadk', 'ltkj', 'lzhs', 'rxyzccj',
    #              'jztxedk', 'jxphdk', 'bmsw', 'tjwdpadk', 'tsjr', 'xxwdk', 'hddk', 'padk2', 'tjtdk', 'jjkjjr', 'zhyx_dk_22912', 'whzhjr',
    #              'srzc4', 'zbdkj', 'gjdk', 'whrtzndk', 'xjfq', 'zxsdk', 'hbjf', 'sdsljf', 'hjdk_test1', 'jydk2', 'fpxdkqz', 'whbyx', 'whclcf',
    #              'dkgxb4', 'srzc', 'kcjrdk', 'yhsw', 'srzc8', 'qddk2', 'rgjr', 'jyjr', 'zcyk', 'hytdk', 'ctmdk', 'szylx', 'zrg', 'yndk', 'hbyy',
    #              'xadk', 'bayrdk', 'ljdk', 'lzxd', 'cfjr', 'ddw', 'zytz', 'nrdk', 'dkqs_53557', 'zrdk', 'xusheng', 'whxydk', 'yxzt', 'tehdk',
    #              'hrsd', 'mchdk', 'wzdk', 'zhmdk', 'zjdk', 'yhxcdk', 'vm_loan_jjl', 'dkgxb3', 'hzyh', 'dcdk', 'jswhjr', 'yyct', 'zjyxph', 'gzpr',
    #              'dkgxb1', 'cxqdk', 'hrjr', 'ylkd', 'bjcyhd', 'yljhdk', 'ljd', 'hldk', 'zddk', 'wwysdk', 'ssr', 'syxdk', 'ymjr', 'wqydk', 'hckg',
    #              'fpxdkxffq', 'zh2yx_dk_81495', 'ddbdk', 'dkgxb5', 'dhjr', 'xdsjr', 'zjpaph', 'chqydk', 'ymdk1', 'lzsdk', 'dsdjr', 'tddk', 'bsyx',
    #              'srzc9', 'jrk', 'lkdk', 'yzjf', 'fojr', 'wgtbjqr', 'fydk', 'hlsddk', 'rybdk', 'yxlzcdk', 'ryd', 'hhxyyd', 'ztxxrdk', 'yxph', 'qzbdk',
    #              'yzxdk', 'paxd', 'zdxdk', 'ksjrdk', 'mdjr', 'dksw', 'dkcjj', 'zyyhdk', 'jmjfdk', 'bhjr', 'kdddk', 'cddk', 'xxwlkj', 'zqjf', 'bjpaphdk',
    #              'cgdk', 'wmdk', 'ddjdfwpt', 'jldk', 'gzlj', 'czjr', 'hyjr', 'paphwh', 'zhjhdk', 'yxjydk', 'nadk', 'jydk5', 'yxph_test', 'shcqwlkj',
    #              'clmdk', 'jnwr', 'zsftwl', 'ymdk_test2', 'pdjrwd', 'wddk', 'dkbs_23019', 'tangdadk', 'ssjxgrd', 'gzxxdk', 'dcjr', 'dkgxb6', 'pljfdk',
    #              'xlxdk', 'ddxd', 'bskjdk2', 'lcydk', 'hjqyd', 'mzgcdk', 'tjjhjfdk', 'njpajtdk', 'jydk4', 'hxph', 'fpxdk', 'jjxydk', 'dkdk', 'drjr',
    #              'lxdk',
    #              'yljr', 'hfdk', 'wyd', 'yxyx_ceshi_dk_51200', 'njpaphdk', 'tcdk', 'bxhs', 'ycxedk', 'zypdk', 'sjjrdk', 'szdk', 'aldk', 'ymmsjd',
    #              'bbdk', 'tjhcddk', 'hzsw', 'tjsxd', 'mdtjdk', 'hljpadk3', 'srzc2', 'fpxdkyksq', 'bbyd', 'szpaph', 'mmqb', 'yxztdk_base_vm', 'whhwdk',
    #              'drwzdk2', 'dmf_52974', 'jjjfdk', 'szydk', 'ntfcdk_base_vm', 'qrdk', 'jycf', 'drsdk', 'hzsw2', 'drwzdk', 'ymdk', 'cs-dk_57829',
    #              'shjfdk', 'dsdk', 'ztzh', 'xhjjr', 'hbxljr', 'njhjjr', 'srzc3', 'jhsw', 'hnxzndk', 'wxjdk', 'bmjf', 'thdk', 'jydk', 'bazx', 'jydk_1',
    #              'zy2dk', 'hjdk', 'tmdk', 'wxhjjf', 'zsyhsdd', 'gyyd2', 'zcsd', 'qddk', 'ybjrdk', 'yxydk', 'gyyd', 'cs_dk_45395', 'hzdk', 'srzc7',
    #              'sdjf', 'yftx', 'jrdk', 'thdk_base_vm', 'fkldk', 'xqwlkj', 'paphdk', 'srzc1', 'whhgjf', 'szhxjr', 'ssjxqyd', 'pxsw', 'adzx', 'stjrdk',
    #              'sxhdk', 'yrdk', 'xmdk2', 'cldk', 'dydk', 'xnjdk', 'dsjrdk', 'zyww', 'lxkj', 'ladk', 'hxdk', 'yx_dk_35717', 'mchappdk',
    #              'zjchdk', 'fpxdkty', 'xmfyhjr', 'wdjr', 'dkgxb7', '4-cs-dk_12351', 'gxpfjr', 'nbyhzjd', 'jkdk', 'gzyes', 'chlpayh', 'gxjrdk',
    #              'ygyd', 'loan_quanzhou_basevm', 'drjfdk', 'cyjdk', 'zyxjyxhs', 'tasddk', 'wtjr', 'jryw', 'zdfq', 'bjwdcfgl', 'bszc', 'hldk2',
    #              'huarui', 'lcxdsp', 'yxyx_ceshi_dk_47311', 'xxgjdk', 'stjj', 'ztxxxdk', 'wdk', 'bajf', 'grhz', 'brzjdk', 'ttd', 'jsjxjc',
    #              'jsyhdk', 'yxldk', 'srzc5', 'xhjf', 'bwldk', 'ddk_48079', 'madk_base_vm', 'hxdkhs', 'dpdbcd', 'hbjsjh', 'xcdk', 'jydk3',
    #              'mmdkhs', 'gzqcwl51gxdk', 'zgyhdk', 'lsdk', 'paph', 'whqcdk', 'xxxxdk', 'srzc6', 'wlxjd', 'axxx', 'ysphdk', 'hjdk_test2',
    #              'whxrh', 'xdyx', 'hljpadk2', 'lsdkhs', 'szcsd', 'qhaljrdk', 'gxkjdk', 'jjhdk', 'gzyhdk', 'rzcs', 'ymdkcs', 'tcd', 'hlqcfmjdk',
    #              'wxph', 'tyjxr', 'jrkdk', 'hnxzn', 'sbjr', 'xdk', 'ryjr', 'bjhxdk', 'dzyd', 'ddsd', 'hswl', 'lgydk', 'hyjrdk', 'mzdk', 'jskd',
    #              'zxyx_ceshi_dk_39099', 'dkgxb0', 'jzjrhs', 'hzbfx', 'dhjr2', 'syzgyhdk', 'pahgdk', 'ydjhgc', 'bszc_back', 'cyqb', 'bm', 'gdlrf',
    #              'qyxd', 'hmjfnbjm', 'jawd', 'fjdk', 'whzcdk', 'scjr', 'wkjdfwpt', 'yfacedk', 'hzhw', 'sxsd', 'jzjr', 'hxjr', 'sxwhydjf', 'wzf',
    #              'dk16_71753', 'zhyjdkj1', 'zhejdkj1', 'jdkj2', 'lhjr', 'dcdkyq', 'zhyjdkj2', 'jdkj', 'zhejdkj2']

    # "dcdkyq",
    #     "dkdk", "dksw", "dsjrdk", "fpxdk", "fpxdk2", "gjdk", "gxkjdk", "gzxxdk", "hddk", "hfdk", "hldk", "hxdk",
    #     "jdkj", "jdkj2", "jjdktz", "jjjfdk", "jjxydk", "jldk", "jrdk", "jrkdk", "jzdk", "kdddk", "ksdk", "lgydk",
    #     "lkdk", "lsdk", "madk_base_vm", "mchdk", "mdtjdk", "nadk", "njpaphdk", "njydkd", "ntfcdk_base_vm", "padk2",
    #     "paphdk", "qbjdk", "qddk", "qhaljrdk", "scdktx", "scdk_yq", "sdjrdk", "shjfdk", "sjjrdk", "stjrdk", "sxydk",
    #     "syzgyhdk", "szksdk_base_vm", "szydk", "tcdk", "tddk", "tdjrdk_base_vm", "thdk", "thdk_base_vm", "tjjhjfdk",
    #     "tjwdpadk", "wdk", "whqcdk", "whrtzndk", "whxrkdk", "wxdk", "xadk", "xadk_src", "xcdk", "xdk", "xhjfdk",
    #     "xlxdk", "xnjdk", "ybjrdk", "ydkdywwh", "yfacedk", "yhxcdk", "yqsdkys", "yrdk", "ysphdk", "yxldk",
    #     "yxlzcdk", "yxydk", "yxztdk_base_vm", "yzxdk", "zbdkj", "zddk", "zgyhdk", "zjchdk", "zxsdk", "zyyhdk","xtw"]
    domain_names = ['phone']

    # cfg_names = cfg_names[0]
    # domain_names = domain_names[0]
    # cfg_names = ['gjdk']
    # cfg_names.remove('gjdk')
    # cfg_names = ['gjdk']
    # domain_names = None

    for cfg_name in cfg_names:
        get_user_sentence_data(
            cfg_name=cfg_name, domain_names=domain_names, save_file_name=dst_dir + 'content.txt')


def get_wav(url="http://global.res.btows.com/data/record/152784226920180601/15562235526_1527842191_1885916k.wav"):
    wavf = ''
    abs_path = os.path.realpath(__file__)
    socket.setdefaulttimeout(5.0)
    if not url.startswith("http"):
        print("%s not http" % url)
    else:
        if not os.path.exists('wav_data'):
            os.makedirs('wav_data')
        try:
            fn0 = url.split('/')[-1]
            wavf = 'wav_data/%s' % fn0
            urllib.request.urlretrieve(url, wavf)
        except Exception as e:
            print("get_wav %s error: " % str(url) +
                  str(type(e).__name__) + ": " + str(e))
    return wavf


def main():
    # test1()
    # get_wav()
    get_user_sentence_data_cfgs()


if __name__ == '__main__':
    main()
