import torch
import re
from apex.normalization.fused_layer_norm import FusedLayerNorm
from mpu.layers import VocabParallelEmbedding
from mpu.transformer import GPT2ParallelSelfAttention
from mpu.transformer import GPT2ParallelMLP
from mpu import copy_to_model_parallel_region, gather_from_model_parallel_region
from mpu import get_model_parallel_world_size

import torch.utils.checkpoint as cp

class CPM():
    def __init__(self, declares, calculations):
        self.declares = declares
        self.calculations = calculations

    def generate_layer_blocks(self):
        self.layers = {}
        for layer in self.declares.split('\n'):
            m = re.search(r'self.layer([0-9]+)', layer)
            layer_id = int(m.group(1))
            self.layers[layer_id] = layer

        self.blocks = [[]]
        for line in self.calculations.split('\n'):
            self.blocks[-1].append(line)
            if '+' in line:
                self.blocks.append([])

    def generate_stage(self, start, end):
        inputs = []
        outputs = []
        declare = []
        calculation = []
        for i in range(start, end):
            calculation.append(self.blocks[i])
            for line in self.blocks[i]:
                m = re.search(r'self.layer([0-9]+)', line)
                if m is not None:
                    layer_id = int(m.group(1))
                    declare.append(self.layers[layer_id])
                out = re.findall(r'out\d+', line)
                for arg in out[1:]:
                    if arg not in outputs and arg not in inputs:
                        inputs.append(arg)
                if out[0] not in outputs:
                    outputs.append(out[0])
        return declare, calculation, inputs, outputs

class Stage(torch.nn.Module):
    def __init__(self, inputs, outputs, declares, calcus, fraction):
        super(Stage, self).__init__()

        # print("{} {} {}".format(inputs, outputs, fraction), flush = True)
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

        exec('\n'.join(declares))

        if len(fraction) > 0:
            assert sum(fraction) == len(calcus)

        if len(fraction) == 1:
            no_cp_ = ["{} = args[{}]".format(name, i) for i, name in enumerate(inputs)]

            cp_ = sum(calcus, [])
            cp_return = []
            for output in outputs:
                if output not in inputs:
                    cp_return.append(output)

            cp_input = ', '.join(inputs)
            cp_return = ', '.join(cp_return)
            no_cp_.append(self.cp_forward(cp_, 0, cp_input + ", dummy", cp_return))
            no_cp_.append("self.__class__.func0 = func0")
            no_cp_.append("%s = cp.checkpoint(self.func0, %s, self.dummy)" % (cp_return, cp_input))

            no_cp_.append("self.out = ({},)".format(', '.join(outputs)))

            self.no_cp = '\n'.join(no_cp_)
        elif len(fraction) == 0:
            self.cp = "assert 1 == 0"
            no_cp_ = sum(calcus, [])

            no_cp_ = ["{} = args[{}]".format(name, i) for i, name in enumerate(inputs)] + no_cp_
            no_cp_.append("self.out = ({},)".format(', '.join(outputs)))

            self.no_cp = '\n'.join(no_cp_)
        else:
            cp_list = []
            start = 0
            for i in fraction:
                cp_list.append(sum(calcus[start:start + i], []))
                start += i

            cp_inputs_list = []
            cp_outputs_list = []
            for cp_ in cp_list:
                cp_inputs = []
                cp_outputs = []
                for line in cp_:
                    out = re.findall(r'out\d+', line)
                    for arg in out[1:]:
                        if arg not in cp_outputs and arg not in cp_inputs:
                            cp_inputs.append(arg)
                    if out[0] not in cp_outputs:
                        cp_outputs.append(out[0])
                cp_inputs_list.append(cp_inputs)
                cp_outputs_list.append(cp_outputs)

            no_cp_ = ["{} = args[{}]".format(name, i) for i, name in enumerate(inputs)]

            for i, cp_ in enumerate(cp_list):
                cp_inputs = cp_inputs_list.pop(0)
                cp_outputs = cp_outputs_list.pop(0)

                required_outputs = set(outputs)
                for inputs in cp_inputs_list:
                    required_outputs |= set(inputs)

                cp_return = []
                for output in required_outputs:
                    if output in cp_outputs:
                        cp_return.append(output)

                cp_input = ', '.join(cp_inputs)
                cp_return = ', '.join(cp_return)

                if i == 0:
                    no_cp_.append(self.cp_forward(cp_, i, cp_input + ", dummy", cp_return))
                    no_cp_.append("self.__class__.func%d = func%d" % (i, i))
                    no_cp_.append("%s = cp.checkpoint(self.func%d, %s, self.dummy)" % (cp_return, i, cp_input))
                else:
                    no_cp_.append(self.cp_forward(cp_, i, cp_input, cp_return))
                    no_cp_.append("self.__class__.func%d = func%d" % (i, i))
                    no_cp_.append("%s = cp.checkpoint(self.func%d, %s)" % (cp_return, i, cp_input))

            no_cp_.append("self.out = ({},)".format(', '.join(outputs)))
            self.no_cp = '\n'.join(no_cp_)

    def forward(self, *args):
        exec(self.no_cp)
        return self.out

    def cp_forward(self, cp, idx, cp_input, cp_return):
        f = "def func%d(self, %s):\n\t" % (idx, cp_input)
        f += '\n\t'.join(cp)
        f += '\n\treturn %s' % cp_return
        return f