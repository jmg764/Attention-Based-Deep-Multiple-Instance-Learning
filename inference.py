{\rtf1\ansi\ansicpg1252\cocoartf2576
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red255\green255\blue255;\red27\green31\blue35;\red27\green31\blue34;
\red109\green109\blue109;}
{\*\expandedcolortbl;;\cssrgb\c100000\c100000\c100000;\cssrgb\c14147\c16111\c18054;\cssrgb\c14118\c16078\c18039;
\cssrgb\c50196\c50196\c50196;}
\margl1440\margr1440\vieww11580\viewh12080\viewkind0
\deftab720

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat2 \trbrdrt\brdrnil \trbrdrl\brdrnil \trbrdrt\brdrnil \trbrdrr\brdrnil 
\clvertalt \clshdrawnil \clwWidth10948\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadl200 \clpadr200 \gaph\cellx4320
\clvertalt \clshdrawnil \clwWidth10948\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadl200 \clpadr200 \gaph\cellx8640
\pard\intbl\itap1\cell
\pard\intbl\itap1\pardeftab720\sl400\partightenfactor0

\f0\fs30 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \
# Licensed to the Apache Software Foundation (ASF) under one\
# or more contributor license agreements.  See the NOTICE file\
# distributed with this work for additional information\
# regarding copyright ownership.  The ASF licenses this file\
# to you under the Apache License, Version 2.0 (the\
# "License"); you may not use this file except in compliance\
# with the License.  You may obtain a copy of the License at\
#\
#   http://www.apache.org/licenses/LICENSE-2.0\
#\
# Unless required by applicable law or agreed to in writing,\
# software distributed under the License is distributed on an\
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\
# KIND, either express or implied.  See the License for the\
# specific language governing permissions and limitations\
# under the License.\
\
from __future__ import print_function\
\
import os\
import torch\
\
# Network definition\
from model_def import Attention\
\
def model_fn(model_dir):\
    print("In model_fn. Model directory is -")\
    print(model_dir)\
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\
    model = Attention()\
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:\
        print("Loading the histopathology mil model")\
        model.load_state_dict(torch.load(f, map_location=device))\
    return model\cell \lastrow\row}