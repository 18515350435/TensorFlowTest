# coding:utf-8

import json
import urllib.parse as parse
from wsgiref.simple_server import make_server


# 定义函数，参数是函数的两个参数，都是python本身定义的，默认就行了。
def application(environ, start_response):
    from io import StringIO
    stdout = StringIO()
    print("Hello world!", file=stdout)
    print(file=stdout)
    h = sorted(environ.items())
    for k, v in h:
        print(k, '=', repr(v), file=stdout)
    params = parse.parse_qs(environ['QUERY_STRING'])
    # 获取get中key为name的值
    name = params['name1']
    no = params['no']

    # 组成一个数组，数组中只有一个字典
    dic = {'name1': name, 'no': no}
    start_response("200 OK", [('Content-Type', 'text/plain; charset=utf-8')])
    return [dic.encode("utf-8")]

    # 定义文件请求的类型和当前请求成功的code
    start_response('200 OK', [('Content-Type', 'text/html')])
    # environ是当前请求的所有数据，包括Header和URL，body，这里只涉及到get
    # 获取当前get请求的所有数据，返回是string类型
    '''
    params = parse.parse_qs(environ['QUERY_STRING'])
    # 获取get中key为name的值
    name = params['name1']
    no = params['no']

    # 组成一个数组，数组中只有一个字典
    dic = {'name1': name, 'no': no}
    start_response("200 OK", [('Content-Type', 'text/plain; charset=utf-8')])
    return ["asdas"]
    '''


if __name__ == "__main__":
    port = 6088
    httpd = make_server("0.0.0.0", port, application)
    print("serving http on port {0}...".format(str(port)))
    httpd.serve_forever()
# ----------------------------POST----------------------------
# # coding:utf-8
#
# import json
# from wsgiref.simple_server import make_server
#
#
# # 定义函数，参数是函数的两个参数，都是python本身定义的，默认就行了。
# def application(environ, start_response):
#     # 定义文件请求的类型和当前请求成功的code
#     start_response('200 OK', [('Content-Type', 'application/json')])
#     # environ是当前请求的所有数据，包括Header和URL，body
#
#     request_body = environ["wsgi.input"].read(int(environ.get("CONTENT_LENGTH", 0)))
#     request_body = json.loads(request_body)
#
#     name = request_body["name"]
#     no = request_body["no"]
#
#     # input your method here
#     # for instance:
#     # 增删改查
#
#     dic = {'myNameIs': name, 'myNoIs': no}
#
#     return [json.dumps(dic)]
#
#
# if __name__ == "__main__":
#     port = 6088
#     httpd = make_server("0.0.0.0", port, application)
#     print("serving http on port {0}...".format(str(port)))
#     httpd.serve_forever()