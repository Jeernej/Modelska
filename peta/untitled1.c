//# -*- coding: utf-8 -*-
//"""
//Created on Mon Nov 27 21:59:50 2017
//
//@author: jernej
//"""


TRandom1*r1=new TRandom1();
TRandom2 *r2=new TRandom2();
TRandom3 *r3=new TRandom3();

for (Int_t i=0; i<500; i++) {Float_t RR =r3->Uniform(0,1); cout << RR << endl; } //print to terminal

ofstream myfile;
myfile.open ("TRandom1_1000000.txt");
myfile << "TS ns nserr\n";
for (Int_t i=0; i<1000000; i++) {Float_t RR =r1->Uniform(0,1); myfile << RR <<"\n";} //write to file
myfile.close();

ofstream myfile;
myfile.open ("TRandom2_1000000.txt");
myfile << "TS ns nserr\n";
for (Int_t i=0; i<1000000; i++) {Float_t RR =r2->Uniform(0,1);  myfile << RR <<"\n";} //write to file
myfile.close();

ofstream myfile;
myfile.open ("TRandom3_1000000.txt");
myfile << "TS ns nserr\n";
for (Int_t i=0; i<1000000; i++) {Float_t RR =r3->Uniform(0,1);  myfile << RR <<"\n";} //write to file


TStopwatch *st=new TStopwatch();

st->Start();
for (Int_t i=0; i<1000000; i++) { Float_t RR =r1->Uniform(0,1); }
st->Stop();
cout << "Random:  " << st->CpuTime() << endl;


st->Start();
for (Int_t i=0; i<1000000; i++) { Float_t RR =r2->Uniform(0,1); }
st->Stop();
cout << "Random:  " << st->CpuTime() << endl;


st->Start();
for (Int_t i=0; i<1000000; i++) { Float_t RR =r3->Uniform(0,1); }
st->Stop();
cout << "Random:  " << st->CpuTime() << endl;


TH1D *h1=new TH1D("h1","TRandom1",50,0,1); // normirana vrednost bi morala bit pri 1/500=0.002
TH1D *h2=new TH1D("h2","TRandom2",50,0,1);
TH1D *h3=new TH1D("h3","TRandom3",50,0,1);

st->Start();
for (Int_t i=0; i<1000000; i++) { h1->Fill(r1->Uniform(0,1)); }
st->Stop();
cout << "Random:  " << st->CpuTime() << endl;


st->Start();
for (Int_t i=0; i<1000000; i++) { h2->Fill(r2->Uniform(0,1)); }
st->Stop();
cout << "Random:  " << st->CpuTime() << endl;


st->Start();
for (Int_t i=0; i<1000000; i++) { h3->Fill(r3->Uniform(0,1)); }
st->Stop();
cout << "Random:  " << st->CpuTime() << endl;

//st->Start();
//for (Int_t i=0; i<500; i++) { for (Int_t j=0; j<1000; j++)  {h4->Fill(i,1);} }
////for (Int_t i=0; i<500; i++) { h4->SetBinContent(i,1000); }
//st->Stop();
//cout << "Random:  " << st->CpuTime() << endl;

//Double_t scale = 1/h1->Integral();
//h1->Scale(scale);

Double_t norm1 = h1->GetEntries();
h1->Scale(1/norm1);
h1-> SetMinimum(0);
h1->SetLineColor(kBlue);


Double_t norm2 = h2->GetEntries();
h2->Scale(1/norm2);
h2 -> SetMinimum(0);
h2->SetLineColor(kRed);


Double_t norm3 = h3->GetEntries();
h3->Scale(1/norm3);
h3 -> SetMinimum(0);
//h3->SetMarkerStyle(21);
h3->SetLineColor(kBlack);

//Double_t norm4 = h4->GetEntries();
//h4->Scale(1/norm4);
//h4 -> SetMinimum(0);
////h4->SetMarkerStyle(21);
//h4->SetLineColor(kGreen);

//THStack *hstack = new THStack("hstack","uniform");
//hstack->Add(h1);
//hstack->Add(h2);
//hstack->Add(h3);
//
//hstack->Draw("nostack");


h1->Draw();
h2->Draw("same");
h3->Draw("same");
//h4->Draw("same");

gPad->BuildLegend();



TF1* konst = new TF1("konst","0.02",0,1);
h3->Chisquare(konst)



c1->Clear();

 // fiting the spectrum
 TF1* func=new TF1("func",function,0,100,4);
 func->SetParameters(1,100,50,2);
 main->Fit("func","QWEMR"); // or with likelihood: "LQWEMR"
 
 // printing out size of line
 Float_t size=func->GetParameter(1);
 cout << "Size of line: " << size << endl;
 
 // store result to histogram 'sum'
 sum->Fill(size);
   }
 
   // showing the results of the fit
   sum->Draw();
 }
 
 // fitting function
 Double_t function(Double_t* x, Double_t* p)
 {
   Double_t dummy=(x[0]-p[2])/p[3];
   Double_t result=p[0] + p[1]*exp(-0.5*dummy*dummy)/ (2.506 * p[3]);
   return result;
 }