# digix服务数据格式说明：
## 请求方法：POST
请求格式:
```
{
	"history" : ["utterance1", "utterance2", "utterance n"]
}

字段说明：
history: list 类型，表示历史对话信息
```
返回格式：
```
{
	"code"：0,
	"message" : "",
	"attrs" : [{
			"name" : "",
			"attrname" : "",
			"attrvalue" : ""
		}, ……
	]
}
字段说明：
code: int 类型，返回状态码， // 返回码，0: 成功
message: string 类型，生成的回复消息
attrs: list 类型，表示生成语句可能用到的知识三元组，可能为空，如果为空，返回值为[]
```