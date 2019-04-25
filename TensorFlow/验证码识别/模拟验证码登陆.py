# coding: UTF-8
#
import urllib.request
import urllib.parse
import re
import shutil
import http.cookiejar

class LoginJust():
    def __init__(self,url,url1,url2,header):
        self.url=url
        self.url1=url1
        self.url2=url2
        self.header=header
        return

#创建opener，包含header信息和cookie
    def CreateOpener(self):
        #实例化cookie对象
        cookie=http.cookiejar.CookieJar()
        #创建一个cookie处理器
        CookieHandle=urllib.request.HTTPCookieProcessor(cookie)
        #创建带有cookie的opener
        opener=urllib.request.build_opener(CookieHandle)
        #传入header
        head=[]
        for key,value in self.header.items():
            elem=(key,value)
            head.append(elem)
        opener.open(self.url)
        return opener

#获取验证码图片，并保存到本地
    def getImage(self,opener):
        path = "imageCode_1.jpg"
        # 带cookie和header
        img = opener.open(self.url2)
        with open(path, "wb") as f:
            shutil.copyfileobj(img, f)
        #print(os.stat(path).st_size, 'characters copied')
        return

#获取后台发送来的字符串
    def getResponse(self,opener):
        getReponse = opener.open(self.url1)
        #
        gRResult = getReponse.read().decode("utf-8")
        return gRResult

#获取post所需的encoded参数
    def achieveCode(self,response):
        # scode、sxh是response中得来
        scode = response.split("#")[0]
        sxh = response.split("#")[1]
        # userAccount、userPassword
        userAccount = self.account
        userPassword =self.pwd
        code = userAccount + "%%%" + userPassword
        # 最终需要获取的post的数据
        encode = ""
        # 进行账号密码加密，获取code的值
        i = 0
        while (i < len(code)):
            if i < 20:
                encode = encode + code[i:i + 1] + scode[0:int(sxh[i])]
                # print(str(i)+"_encode:"+encode)
                scode = scode[int(sxh[i]):len(scode)]
                # print(str(i)+"_scode:"+scode)
            else:
                encode = encode + code[i:len(code)]
                # print(str(i)+"_here_encode"+encode)
                i = len(code)
            i = i + 1
        return encode

#判断登陆是否成功
    def IsLoginS(self):
#验证码的值
        codeValue=input("请输入验证码：")
        data={"referer":"","userid":18515350435,"password":"zhangpeihan789","captcha":codeValue}
#把data转换为post的数据格式
        postData=urllib.parse.urlencode(data)
        result=opener.open(self.url,postData.encode("utf-8"))
        Result=result.read().decode("utf-8")
        regex=re.compile(r"1440407133")
        #print(type(regex.search(Result)))
        if regex.search(Result)!=None:
            return True
        else:
            print("Someting Error happened:登陆失败")
        return False

#再次请求验证码


# 需要post的网址的URL
url = "http://www.sjgtw.com/login"
# 获取后台数据的网址
url1 = "http://jwgl.just.edu.cn:8080/Logon.do?method=logon&flag=sess"
# 获取验证码图片的地址
url2 = "http://www.sjgtw.com/generateImage"
#header
header=\
    {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36",
        "Referer":"http://www.sjgtw.com",
        "Host":"http://www.sjgtw.com",
        "Connection":"keep-alive",
        "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding":"gzip, deflate",
        "Accept-Language":"zh-CN,zh;q=0.8",
        "Upgrade-Insecure-Requests":"1",
        "Cache-Control":"max-age=0",
    }
# account=input("请输入用户名：")
# pwd=input("请输入密码：")
#实例化对象
lj=LoginJust(url,url1,url2,header)
opener=lj.CreateOpener()
lj.getImage(opener)
# encoded=lj.achieveCode(lj.getResponse(opener))
if lj.IsLoginS():
    print("登陆成功")