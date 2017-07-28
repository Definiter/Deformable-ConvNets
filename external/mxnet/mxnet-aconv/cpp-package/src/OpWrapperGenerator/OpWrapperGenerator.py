﻿# -*- coding: utf-8 -*-
from ctypes import *
from ctypes.util import find_library
import os
import logging
import platform
import re
import sys
import tempfile

class EnumType:
    name = ''
    enumValues = []
    def __init__(self, typeName = 'ElementWiseOpType', \
                 typeString = "{'avg', 'max', 'sum'}"):
        self.name = typeName
        if (typeString[0] == '{'):  # is a enum type
            isEnum = True
            # parse enum
            self.enumValues = typeString[typeString.find('{') + 1:typeString.find('}')].split(',')
            for i in range(0, len(self.enumValues)):
                self.enumValues[i] = self.enumValues[i].strip().strip("'")
        else:
            logging.warn("trying to parse none-enum type as enum: %s" % typeString)
    def GetDefinitionString(self, indent = 0):
        indentStr = ' ' * indent
        ret = indentStr + 'enum class %s {\n' % self.name
        for i in range(0, len(self.enumValues)):
            ret = ret + indentStr + '  %s = %d' % (self.enumValues[i], i)
            if (i != len(self.enumValues) -1):
                ret = ret + ","
            ret = ret + "\n"
        ret = ret + "};\n"
        return ret
    def GetDefaultValueString(self, value = ''):
        return self.name + "::" + value
    def GetEnumStringArray(self, indent = 0):
        indentStr = ' ' * indent
        ret = indentStr + 'static const char *%sValues[] = {\n' % self.name
        for i in range(0, len(self.enumValues)):
            ret = ret + indentStr + '  "%s"' % self.enumValues[i]
            if (i != len(self.enumValues) -1):
                ret = ret + ","
            ret = ret + "\n"
        ret = ret + indentStr + "};\n"
        return ret
    def GetConvertEnumVariableToString(self, variable=''):
        return "%sValues[int(%s)]" % (self.name, variable)


class Arg:
    typeDict = {'boolean':'bool',\
        'Shape(tuple)':'Shape',\
        'Symbol':'Symbol',\
        'NDArray':'Symbol',\
        'NDArray-or-Symbol':'Symbol',\
        'Symbol[]':'const std::vector<Symbol>&',\
        'Symbol or Symbol[]':'const std::vector<Symbol>&',\
        'NDArray[]':'const std::vector<Symbol>&',\
        'caffe-layer-parameter':'::caffe::LayerParameter',\
        'NDArray-or-Symbol[]':'const std::vector<Symbol>&',\
        'float':'mx_float',\
        'real_t':'mx_float',\
        'int':'int',\
        'int (non-negative)': 'uint32_t',\
        'long (non-negative)': 'uint64_t',\
        'int or None':'dmlc::optional<int>',\
        'long':'int64_t',\
        'double':'double',\
        'string':'const std::string&'}
    name = ''
    type = ''
    description = ''
    isEnum = False
    enum = None
    hasDefault = False
    defaultString = ''
    def __init__(self, opName = '', argName = '', typeString = '', descString = ''):
        self.name = argName
        self.description = descString
        if (typeString[0] == '{'):  # is enum type
            self.isEnum = True
            self.enum = EnumType(self.ConstructEnumTypeName(opName, argName), typeString)
            self.type = self.enum.name
        else:
            try:
                self.type = self.typeDict[typeString.split(',')[0]]
            except:
                print 'argument "%s" of operator "%s" has unknown type "%s"' % (argName, opName, typeString)
                pass
        if typeString.find('default=') != -1:
            self.hasDefault = True
            self.defaultString = typeString.split('default=')[1].strip().strip("'")
            if typeString.startswith('string'):
                self.defaultString = self.MakeCString(self.defaultString)
            elif self.isEnum:
                self.defaultString = self.enum.GetDefaultValueString(self.defaultString)
            elif self.defaultString == 'None':
                self.defaultString = self.type + '()'
            elif self.defaultString == 'False':
                self.defaultString = 'false'
            elif self.defaultString == 'True':
                self.defaultString = 'true'
            elif self.defaultString[0] == '(':
                self.defaultString = 'Shape' + self.defaultString
            elif self.type == 'dmlc::optional<int>':
                self.defaultString = self.type + '(' + self.defaultString + ')'
            elif typeString.startswith('caffe-layer-parameter'):
                self.defaultString = 'textToCaffeLayerParameter(' + self.MakeCString(self.defaultString) + ')'
                hasCaffe = True

    def MakeCString(self, str):
        str = str.replace('\n', "\\n")
        str = str.replace('\t', "\\t")
        return '\"' + str + '\"'

    def ConstructEnumTypeName(self, opName = '', argName = ''):
        a = opName[0].upper()
        # format ArgName so instead of act_type it returns ActType
        argNameWords = argName.split('_')
        argName = ''
        for an in argNameWords:
            argName = argName + an[0].upper() + an[1:]
        typeName = a + opName[1:] + argName
        return typeName

class Op:
    name = ''
    description = ''
    args = []

    def __init__(self, name = '', description = '', args = []):
        self.name = name
        self.description = description
        # add a 'name' argument
        nameArg = Arg(self.name, \
                      'symbol_name', \
                      'string', \
                      'name of the resulting symbol')
        args.insert(0, nameArg)
        # reorder arguments, put those with default value to the end
        orderedArgs = []
        for arg in args:
            if not arg.hasDefault:
                orderedArgs.append(arg)
        for arg in args:
            if arg.hasDefault:
                orderedArgs.append(arg)
        self.args = orderedArgs

    def WrapDescription(self, desc = ''):
        ret = []
        sentences = desc.split('.')
        lines = desc.split('\n')
        for line in lines:
          line = line.strip()
          if len(line) <= 80:
            ret.append(line.strip())
          else:
            while len(line) > 80:
              pos = line.rfind(' ', 0, 80)+1
              if pos <= 0:
                pos = line.find(' ')
              if pos < 0:
                pos = len(line)
              ret.append(line[:pos].strip())
              line = line[pos:]
        return ret

    def GenDescription(self, desc = '', \
                        firstLineHead = ' * \\brief ', \
                        otherLineHead = ' *        '):
        ret = ''
        descs = self.WrapDescription(desc)
        ret = ret + firstLineHead
        if len(descs) == 0:
          return ret.rstrip()
        ret = (ret + descs[0]).rstrip() + '\n'
        for i in range(1, len(descs)):
            ret = ret + (otherLineHead + descs[i]).rstrip() + '\n'
        return ret

    def GetOpDefinitionString(self, use_name, indent=0):
        ret = ''
        indentStr = ' ' * indent
        # define enums if any
        for arg in self.args:
            if arg.isEnum and use_name:
                # comments
                ret = ret + self.GenDescription(arg.description, \
                                        '/*! \\breif ', \
                                        ' *        ')
                ret = ret + " */\n"
                # definition
                ret = ret + arg.enum.GetDefinitionString(indent) + '\n'
        # create function comments
        ret = ret + self.GenDescription(self.description, \
                                        '/*!\n * \\breif ', \
                                        ' *        ')
        for arg in self.args:
            if arg.name != 'symbol_name' or use_name:
                ret = ret + self.GenDescription(arg.name + ' ' + arg.description, \
                                        ' * \\param ', \
                                        ' *        ')
        ret = ret + " * \\return new symbol\n"
        ret = ret + " */\n"
        # create function header
        declFirstLine = indentStr + 'inline Symbol %s(' % self.name
        ret = ret + declFirstLine
        argIndentStr = ' ' * len(declFirstLine)
        arg_start = 0 if use_name else 1
        if len(self.args) > arg_start:
            ret = ret + self.GetArgString(self.args[arg_start])
        for i in range(arg_start+1, len(self.args)):
            ret = ret + ',\n'
            ret = ret + argIndentStr + self.GetArgString(self.args[i])
        ret = ret + ') {\n'
        # create function body
        # if there is enum, generate static enum<->string mapping
        for arg in self.args:
            if arg.isEnum:
                ret = ret + arg.enum.GetEnumStringArray(indent + 2)
        # now generate code
        ret = ret + indentStr + '  return Operator(\"%s\")\n' % self.name
        for arg in self.args:   # set params
            if arg.type == 'Symbol' or \
                arg.type == 'const std::string&' or \
                arg.type == 'const std::vector<Symbol>&':
                continue
            v = arg.name
            if arg.isEnum:
                v = arg.enum.GetConvertEnumVariableToString(v)
            ret = ret + indentStr + ' ' * 11 + \
                '.SetParam(\"%s\", %s)\n' % (arg.name, v)
        #ret = ret[:-1]  # get rid of the last \n
        symbols = ''
        inputAlreadySet = False
        for arg in self.args:   # set inputs
            if arg.type != 'Symbol':
                continue
            inputAlreadySet = True
            #if symbols != '':
            #    symbols = symbols + ', '
            #symbols = symbols + arg.name
            ret = ret + indentStr + ' ' * 11 + \
                '.SetInput(\"%s\", %s)\n' % (arg.name, arg.name)
        for arg in self.args:   # set input arrays vector<Symbol>
            if arg.type != 'const std::vector<Symbol>&':
                continue
            if (inputAlreadySet):
                logging.error("op %s has both Symbol[] and Symbol inputs!" % self.name)
            inputAlreadySet = True
            symbols = arg.name
            ret = ret + '(%s)\n' % symbols
        ret = ret + indentStr + ' ' * 11
        if use_name:
            ret = ret + '.CreateSymbol(symbol_name);\n'
        else:
            ret = ret + '.CreateSymbol();\n'
        ret = ret + indentStr + '}\n'
        return ret

    def GetArgString(self, arg):
        ret = '%s %s' % (arg.type, arg.name)
        if arg.hasDefault:
            ret = ret + ' = ' + arg.defaultString
        return ret


def ParseAllOps():
    """
    MXNET_DLL int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                                   AtomicSymbolCreator **out_array);

    MXNET_DLL int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                              const char **name,
                                              const char **description,
                                              mx_uint *num_args,
                                              const char ***arg_names,
                                              const char ***arg_type_infos,
                                              const char ***arg_descriptions,
                                              const char **key_var_num_args);
    """
    cdll.libmxnet = cdll.LoadLibrary(sys.argv[1])
    ListOP = cdll.libmxnet.MXSymbolListAtomicSymbolCreators
    GetOpInfo = cdll.libmxnet.MXSymbolGetAtomicSymbolInfo
    ListOP.argtypes=[POINTER(c_int), POINTER(POINTER(c_void_p))]
    GetOpInfo.argtypes=[c_void_p, \
        POINTER(c_char_p), \
        POINTER(c_char_p), \
        POINTER(c_int), \
        POINTER(POINTER(c_char_p)), \
        POINTER(POINTER(c_char_p)), \
        POINTER(POINTER(c_char_p)), \
        POINTER(c_char_p), \
        POINTER(c_char_p)
        ]

    nOps = c_int()
    opHandlers = POINTER(c_void_p)()
    r = ListOP(byref(nOps), byref(opHandlers))
    ret = ''
    ret2 = ''
    for i in range(0, nOps.value):
        handler = opHandlers[i]
        name = c_char_p()
        description = c_char_p()
        nArgs = c_int()
        argNames = POINTER(c_char_p)()
        argTypes = POINTER(c_char_p)()
        argDescs = POINTER(c_char_p)()
        varArgName = c_char_p()
        return_type = c_char_p()

        GetOpInfo(handler, byref(name), byref(description), \
            byref(nArgs), byref(argNames), byref(argTypes), \
            byref(argDescs), byref(varArgName), byref(return_type))

        if name.value[0]=='_':     # get rid of functions like __init__
            continue

        args = []

        for i in range(0, nArgs.value):
            arg = Arg(name.value,
                      argNames[i],
                      argTypes[i],
                      argDescs[i])
            args.append(arg)

        op = Op(name.value, description.value, args)

        ret = ret + op.GetOpDefinitionString(True) + "\n"
        ret2 = ret2 + op.GetOpDefinitionString(False) + "\n"
    return ret + ret2

if __name__ == "__main__":
    #et = EnumType(typeName = 'MyET')
    reload(sys)
    sys.setdefaultencoding('UTF8')
    #print(et.GetDefinitionString())
    #print(et.GetEnumStringArray())
    #arg = Arg()
    #print(arg.ConstructEnumTypeName('SoftmaxActivation', 'act_type'))
    #arg = Arg(opName = 'FullConnected', argName='act_type', \
    #    typeString="{'elu', 'leaky', 'prelu', 'rrelu'},optional, default='leaky'", \
    #    descString='Activation function to be applied.')
    #print(arg.isEnum)
    #print(arg.defaultString)
    #arg = Arg("fc", "alpha", "float, optional, default=0.0001", "alpha")
    #decl = "%s %s" % (arg.type, arg.name)
    #if arg.hasDefault:
    #    decl = decl + "=" + arg.defaultString
    #print(decl)

    temp_file_name = ""
    output_file = '../../include/mxnet-cpp/op.h'
    try:
        # generate file header
        patternStr = ("/*!\n"
                      "*  Copyright (c) 2016 by Contributors\n"
                      "* \\file op.h\n"
                      "* \\brief definition of all the operators\n"
                      "* \\author Chuntao Hong, Xin Li\n"
                      "*/\n"
                      "\n"
                      "#ifndef CPP_PACKAGE_INCLUDE_MXNET_CPP_OP_H_\n"
                      "#define CPP_PACKAGE_INCLUDE_MXNET_CPP_OP_H_\n"
                      "\n"
                      "#include <string>\n"
                      "#include <vector>\n"
                      "#include \"mxnet-cpp/base.h\"\n"
                      "#include \"mxnet-cpp/shape.h\"\n"
                      "#include \"mxnet-cpp/op_util.h\"\n"
                      "#include \"mxnet-cpp/operator.h\"\n"
                      "#include \"dmlc/optional.h\"\n"
                      "\n"
                      "namespace mxnet {\n"
                      "namespace cpp {\n"
                      "\n"
                      "%s"
                      "} //namespace cpp\n"
                      "} //namespace mxnet\n"
                      "#endif  // CPP_PACKAGE_INCLUDE_MXNET_CPP_OP_H_\n")

        # Generate a temporary file name
        tf = tempfile.NamedTemporaryFile()
        temp_file_name = tf.name
        tf.close()
        with open(temp_file_name, 'w') as f:
            f.write(patternStr % ParseAllOps())

    except Exception, e:
      os.remove(output_file)
      if len(temp_file_name) > 0:
        os.remove(temp_file_name)
      raise(e)

    os.system('./move-if-change.sh ' + temp_file_name + ' ' + output_file)
    pass

