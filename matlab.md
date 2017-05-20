# 语法
## 函数 function
```
function [returnv1,returnv2,returnv3] = myfunc(input1,input2)
returnv1 = 'abc';
returnv2 = input1 * input2;
returnv3.name = 1;
returnv3.age = '100';
end
```

## 结构体 structure
不用定义,直接用
```
mystructure.ss = 5;
mystructure.a = 'abc';
```

# 调试
dbstop if error
如果运行出现错误，matlab会自动停在出错的那行，并且保存所有相关变量
