# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:42:15 2016

@author: dtward
"""


class Base:
    def sayHi(self):
        print('Hi')
    def __call__(self):
        self.sayHi()
        
b = Base()
b()

class Derived:
    def sayHi(self):
        print('Hi, I\'m derived')
d = Derived()
d()

# okay so we cannot use the base's special method like this