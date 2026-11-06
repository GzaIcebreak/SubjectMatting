#include "pipeline_q.h"
#include <iostream>
using namespace sm;
int main(){int failures=0;auto check=[&](bool b,const char*n){std::cout<<(b?"PASS ":"FAIL ")<<n<<"\n";failures+=!b;};
ImageU8 im(100,100,3);std::fill(im.data.begin(),im.data.end(),255);
for(int y=20;y<80;y++)for(int x=20;x<80;x++)if(x<30||x>=70||y<30||y>=70)for(int c=0;c<3;c++)im.data[(y*100+x)*3+c]=0;
for(int x=5;x<8;x++)for(int c=0;c<3;c++)im.data[(5*100+x)*3+c]=0;
auto pieces=key_candidates(im,{255,255,255});Mask m(100,100);for(auto&p:pieces)for(size_t i=0;i<m.data.size();i++)m.data[i]|=p.data[i];check(!m.at(50,50),"preserve enclosed hole");check(m.at(6,5),"preserve small meaningful component");
Instances inst;inst.w=10;inst.h=10;inst.scores={.9f,.8f};inst.probs.resize(2,std::vector<float>(100));for(int i=0;i<60;i++)inst.probs[0][i]=1;for(int i=20;i<80;i++)inst.probs[1][i]=1;check(drop_duplicates(inst,{0,1}).size()==2,"retain complementary candidates");inst.probs[1]=inst.probs[0];check(drop_duplicates(inst,{0,1}).size()==1,"remove identical candidate");Mask thin(40,40);for(int y=10;y<30;y++)for(int x=10;x<25;x++)thin.data[y*40+x]=1;for(int x=25;x<35;x++)thin.data[20*40+x]=1;auto t=make_trimap_q(thin);check(t[20*40+30]==255,"thin branch has foreground constraint");auto ht=make_trimap_q(m);check(ht[50*100+50]==0,"interior hole remains background in trimap");return failures?1:0;}
