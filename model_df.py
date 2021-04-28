{\rtf1\ansi\ansicpg1252\cocoartf2576
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red255\green255\blue255;\red27\green31\blue35;\red27\green31\blue34;
\red109\green109\blue109;}
{\*\expandedcolortbl;;\cssrgb\c100000\c100000\c100000;\cssrgb\c14147\c16111\c18054;\cssrgb\c14118\c16078\c18039;
\cssrgb\c50196\c50196\c50196;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat2 \trbrdrt\brdrnil \trbrdrl\brdrnil \trbrdrt\brdrnil \trbrdrr\brdrnil 
\clvertalt \clshdrawnil \clwWidth9358\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadl200 \clpadr200 \gaph\cellx4320
\clvertalt \clshdrawnil \clwWidth9358\clftsWidth3 \clbrdrt\brdrnil \clbrdrl\brdrnil \clbrdrb\brdrnil \clbrdrr\brdrnil \clpadl200 \clpadr200 \gaph\cellx8640
\pard\intbl\itap1\cell
\pard\intbl\itap1\pardeftab720\sl400\partightenfactor0

\f0\fs28 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 # Licensed to the Apache Software Foundation (ASF) under one\
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
import torch\
import torch.nn.functional as F\
import torch.nn as nn\
\
class Attention(nn.Module):\
  def __init__(self):\
    super(Attention, self).__init__()\
    self.L = 512 # 512 node fully connected layer\
    self.D = 128 # 128 node attention layer\
    self.K = 1\
\
    self.feature_extractor_part1 = nn.Sequential(\
        nn.Conv2d(3, 36, kernel_size=5),\
        nn.ReLU(),\
        nn.MaxPool2d(2, stride=2),\
        nn.Conv2d(36, 48, kernel_size=3),\
        nn.ReLU(),\
        nn.MaxPool2d(2, stride=2),\
    )\
\
    self.feature_extractor_part2 = nn.Sequential(\
        nn.Linear(12 * 48 * 30 * 30, self.L),\
        nn.ReLU(),\
        nn.Dropout(),\
        nn.Linear(self.L, self.L),\
        nn.ReLU(),\
        nn.Dropout(),\
    )\
\
    self.attention = nn.Sequential(\
        nn.Linear(self.L, self.D),\
        nn.Tanh(),\
        nn.Linear(self.D, self.K)\
    )\
\
    self.classifier = nn.Sequential(\
        nn.Linear(self.L * self.K, 1),\
        nn.Sigmoid()\
    )\
\
  def forward(self, x):\
    x = x.squeeze(0)\
\
    H = self.feature_extractor_part1(x)\
    H = H.view(-1, 12 * 48 * 30 * 30)\
    H = self.feature_extractor_part2(H)\
\
    A = self.attention(H) # NxK\
    A = torch.transpose(A, 1, 0) # KxN\
    print(A)\
    print(type(A))\
    A = F.softmax(A, dim=1) # softmax over N\
\
    M = torch.mm(A, H)\
\
    Y_prob = self.classifier(M)\
    Y_hat = torch.ge(Y_prob, 0.5).float()\
\
    return Y_prob, Y_hat, A.byte()\
\
  def calculate_classification_error(self, X, Y):\
    Y = Y.float()\
    _, Y_hat, _ = self.forward(X)\
    error = 1. - Y_hat.eq(Y).cpu().float().mean().data\
\
    return error, Y_hat\
\
  def calculate_objective(self, X, Y):\
    Y = Y.float()\
    Y_prob, _, A = self.forward(X)\
    Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)\
    neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))\
\
    return neg_log_likelihood, A\cell \lastrow\row}