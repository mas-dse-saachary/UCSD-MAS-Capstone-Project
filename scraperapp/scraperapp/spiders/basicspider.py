
import scrapy


class BnbSpider(scrapy.Spider):
    name = "math103a"
    
    def start_requests(self):
        urls = ['https://math.ucsd.edu/~alpelayo/Math103A_Winter18.html']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'math103a-%s.html' % page
        
        for content in response.css("body"):
            par=content.css("p::text").extract()
            print dict(par=par)
          

            """content = body.css('p').extract()
            dictionary=dict(content)"""
        
        """with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)"""
