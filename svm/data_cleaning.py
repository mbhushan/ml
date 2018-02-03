#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 05:34:05 2018

@author: manib
"""

import csv

def transform_data():
    filepath = 'data/phising.txt'
    out_file = open('data/phising.csv', 'w')
    csvwriter = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    # index = 0
    row = []
    with open(filepath) as f:
        for line in f:
            row = line.split(' ')
            row = transform_row(row)
            csvwriter.writerow(row)
            
            

def transform_row(row):
    final_row = []
    values = {}
    for i in range(1, len(row)):
        token = row[i].split(':')
        values[int(token[0])] = float(token[1])
    print (values)
    for i in range(68):
        if (i+1) in values:
            final_row.append(values[(i+1)])
        else:
            final_row.append(0)
    final_row.append(int(row[0]))
    # print(final_row)
    return final_row
        


def main():
    transform_data()

if __name__ == '__main__':
    main()