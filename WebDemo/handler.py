import json
import logging
from json import JSONDecodeError

import simplejson
import tornado.web

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__file__)


def func(body):
    return dict(code=0, message="", attrs=[], response_des="success")


class MainHandler(tornado.web.RequestHandler):

    @staticmethod
    def check_param(body):
        if not body or not isinstance(body, str):
            return False
        return True

    @staticmethod
    def is_valid(body):
        try:
            json.loads(body)
        except JSONDecodeError as e:
            logger.info("body is {} json decode error, {}".format(body, e))
            return False
        return True

    def post(self):
        # 获取请求参数
        body = self.request.body.decode("utf-8")
        logger.info("body info is {}".format(body))
        # 请求参数校验
        if not self.check_param(body) or not self.is_valid(body):
            self.write(simplejson.dumps(
                dict(code=105, message="", attrs=[], response_des="input parameter is not correct"),
                ensure_ascii=False))
            return
        data = json.loads(body)
        if not isinstance(data, dict) or not data.get("history"):
            self.write(simplejson.dumps(dict(code=105, message="", attrs=[], response_des="history is none"),
                                        ensure_ascii=False))
            return
        # TODO 业务逻辑代码
        result = func(body)

        # 返回结果报文
        self.write(simplejson.dumps(result, ensure_ascii=False))
