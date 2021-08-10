const url = require('url');
const http = require('http');
const formidable = require('formidable');
const fs = require('fs');
const csv = require('csv-parser');
const { networkInterfaces } = require('os');
const express = require('express');
const connect = require('connect');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const nets = networkInterfaces();
const results = [];

const csvWriter = createCsvWriter({
  path: 'out.csv',
  header: [
    {id: 'match', title: '-1'},
    {id: 'alliance', title: 'NULL'},
    {id: 'tNum', title: '0'},
    {id: 'aScore', title: '0'},
    {id: 'tScore', title: '0'},
  ]
})

var ip;
for (const name of Object.keys(nets)) {
  for (const net of nets[name]) {
    if (net.family ==='IPv4' && !net.internal) {
      if (!results[name]) {
        results[name] = [];
      }
      results[name].push(net.address);
    }
  }
}
ip = results['Wi-Fi'][0];
//console.log(results);
//console.log(results['Wi-Fi'][0]);

//creates an http server
const server = http.createServer(function (request, response) {

  //if the upload file button has been hit
  if (request.method == 'POST') {
    //console.log('POST')
    var body = ''
    request.on('data', function(data) {
      body += data
      //console.log('Partial body: ' + body)
    })
    request.on('end', function() {
      console.log(body);
      var dataArray = body.split(',');
      const csvData = [
        {
          match: dataArray[0],
          alliance: dataArray[1],
          tNum: dataArray[2],
          aScore: dataArray[3],
          tScore: dataArray[4]
        }
      ];
      csvWriter.writeRecords(csvData).then(()=> console.log('CSV successfully written'));
      response.writeHead(200, {'Content-Type': 'text/html'})
      response.end('post received')
    })

  } else {
    //console.log('Get')
    var html = `
            <html>
                <body>
                    <form method="post" action="http://localhost:8080">Name:
                    </form>
                </body>
            </html>`
    response.writeHead(200, {'Content-Type': 'text/html'})
    response.end(html)
  }
})
const port = 8080
const host = ip;
server.listen(port, host)
console.log(`Listening at http://${host}:${port}`)
