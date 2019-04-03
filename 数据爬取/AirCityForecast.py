import requests
from lxml import etree
import time


baseUrl='http://106.37.208.228:8082'

headers = {
    'Cookie': 'UM_distinctid=165dc2b446d0-042fb335371265-37664109-1fa400-165dc2b446e254; CNZZDATA1254743953=669111560-1536995972-%7C1536996102; followcity=54511%2C58367%2C59493%2C57516',
    'Host': '106.37.208.228:8082',
    'Referer': 'http://106.37.208.228:8082/',
    'If-None-Match': '5b9c4fe9-7369',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'
}


def getCityList():
    citys = []
    con = requests.get(baseUrl+'/CityForecast/Index',headers=headers).content.decode('utf-8')
    html = etree.HTML(con)
    cityList = html.xpath('//a[@class="city_item"]')
    for city in cityList:
        citys.append(city.get('data-id'))
    return citys

def getAirCityForecast(city):
    data = []
    param = {
        'CityCode':city
    }
    con = requests.post(baseUrl+'/CityForecast',headers=headers,data=param).content.decode('utf-8')
    # print(con)
    html = etree.HTML(con)
    hourAqiDiv = html.xpath('//div[@class="hourAqiDiv"]')
    if len(hourAqiDiv)<=0:
        return
    for f in hourAqiDiv:
        date = f.xpath('.//div[@class="aqi_title"]/p/text()')[0]
        aqiMin = f.xpath('.//p[@class="aqi_value_number"]/span[1]/text()')[0]
        aqiMax = f.xpath('.//p[@class="aqi_value_number"]/span[2]/text()')[0]
        quality = f.xpath('.//p[@class="aqi_value_level"]/text()')[0]
        primaryPollutant = str(f.xpath('.//p[@class="first_pollutant"]/text()')[0]+f.xpath('.//p[@class="first_pollutant"]/sub/text()')[0]).replace('首要污染物:','')
        print(aqiMin,aqiMax,quality,primaryPollutant,date)
        print('*'*30)
        



if __name__ == '__main__':
    citys = getCityList()# 获取所有的城市
    if len(citys)>0:
        for city in citys:
            getAirCityForecast('110000')#根据城市编码获取空气质量预报
            break




