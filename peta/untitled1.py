# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:59:50 2017

@author: jernej
"""


TRandom1 *r1=new TRandom1();
TRandom2 *r2=new TRandom2();
TRandom3 *r3=new TRandom3();

TStopwatch *st=new TStopwatch();

TH1D *h1=new TH1D("h1","TRandom1",500,0,1);
TH1D *h2=new TH1D("h2","TRandom2",500,0,1);
TH1D *h3=new TH1D("h3","TRandom3",500,0,1);

st->Start();
for (Int_t i=0; i<500; i++) // { h1->Fill(r1->Uniform(0,1)); }
st->Stop();
cout << "Random:  " << st->CpuTime() << endl;

#
#st->Start();
#for (Int_t i=0; i<500000; i++) { h2->Fill(r2->Uniform(0,1)); }
#st->Stop();
#cout << "Random:  " << st->CpuTime() << endl;
#
#
#st->Start();
#for (Int_t i=0; i<500000; i++) { h3->Fill(r3->Uniform(0,1)); }
#st->Stop();
#cout << "Random:  " << st->CpuTime() << endl;


Double_t norm1 = h1->GetEntries();
h1->Scale(1/norm1);
h1-> SetMinimum(0);

Double_t norm2 = h2->GetEntries();
h2->Scale(1/norm2);
h2 -> SetMinimum(0);

Double_t norm3 = h3->GetEntries();
h3->Scale(1/norm3);
h3 -> SetMinimum(0);

h1->Draw();
h2->Draw();
h3->Draw();