import logging

import tornado.httpserver
import tornado.ioloop
import tornado.web

from WebDemo.config import Config
from WebDemo.handler import MainHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__file__)


def main():
    app = tornado.web.Application(handlers=[
        (r'/predict', MainHandler),
    ])
    server = tornado.httpserver.HTTPServer(app)
    port = int(Config.server_port)
    server.bind(port)
    server.start()
    logger.info('digix server started... listening port: %s' % port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
