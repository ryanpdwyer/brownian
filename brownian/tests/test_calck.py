# -*- coding: utf-8 -*-
"""Tests for the _calck.py module"""
import unittest
from nose.tools import eq_
from brownian._calck import img2uri, calck, cli, file_extension

def test_file_extension():
    filename_exp_output = {
        'data.csv': 'csv',
        'data.tar.gz': 'gz',
        '.vimrc': '',
        'Makefile': ''}

    for filename, exp_output in filename_exp_output.viewitems():
        eq_(file_extension(filename), exp_output)


class TestImg2Uri(unittest.TestCase):
    pass