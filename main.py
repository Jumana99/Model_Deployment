# if __name__ == '__main__':
import socketserver
from http.server import BaseHTTPRequestHandler
import deepLearningMdels


class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self, string):
        print("Client Address: {}".format(self.client_address))
        print("Client request: {}:{}{}".format(host, port, self.path))
        self.send_response(200)
        result = deepLearningMdels.execute(string)
        file = bytes(result, 'UTF-8')
        self.end_headers()
        self.wfile.write(file)


host, port = "0.0.0.0", 9000
handler = socketserver.TCPServer((host, port), MyHandler)
print("Server is ready to receive at: {}:{}".format(host, port))
handler.serve_forever()
