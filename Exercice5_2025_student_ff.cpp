#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string.h>
#include <numeric>
#include "ConfigFile.tpp"
#include <algorithm>

using namespace std;

struct couple{
	double x1;
	double x2;
};
	
double energy(const std::vector<double>& fnow, double dx) {
  double ener = 0.0; // TODO: compute quantity E 
  for(auto el: fnow){
	  ener+=el*el;
  }
  return dx*ener;
}
double energy_fac(const std::vector<std::vector<double>>& fnow, double dx) {
  double ener = 0.0; // TODO: compute quantity E 
  for(auto el: fnow){
	  for(auto val : el){
		  ener+=val*val;
	  }
  }
  return dx*dx*ener;
}


void boundary_condition(vector<double> &fnext, vector<double> &fnow, double const& A, double om, \
		double const& t,double const& dt, \
		vector<double> &beta2, string &bc_l, string &bc_r, int &N)
{
      if (bc_l == "fixe"){
        fnext[0] = 0.0; 
	// NB: on peut aussi utiliser la condition "excitation" et poser A=0
      }else if(bc_l == "libre"){
        fnext[0] = fnext[1]; // TODO : Modifier pour imposer la condition au bord gauche libre
      }else if (bc_l =="sortie"){
        fnext[0] =  fnow[0]+sqrt(beta2[0])*(fnow[1]-fnow[0]); // TODO : Modifier pour imposer la condition au bord gauche "sortie de l'onde"
      }else if (bc_l == "excitation"){
        fnext[0] = A*sin(om*t); // TODO : Modifier pour imposer la condition au bord gauche sinusoidale
      }else{
        cerr << "Merci de choisir une condition aux bord gauche valide" << endl;
      }
	      
      if (bc_r == "fixe"){
        fnext[N-1] = 0.0; 
	// NB: on peut aussi utiliser la condition "excitation" et poser A=0	
      }else if(bc_r == "libre"){
        fnext[N-1] = fnext[N-2]; // TODO : Modifier pour imposer la condition au bord droit libre
      }else if (bc_r =="sortie"){
        fnext[N-1] = fnow[N-1]-sqrt(beta2[N-1])*(fnow[N-1]-fnow[N-2]); // TODO : Modifier pour imposer la condition au bord droit "sortie de l'onde"
      }else if (bc_r == "excitation"){ 
        fnext[N-1] = A*sin(om*t); // TODO : Modifier pour imposer la condition au bord droit sinusoidale
      }else{
        cerr << "Merci de choisir une condition aux bord droit valide" << endl;
      }
}

void boundary_condition_fac(vector<vector<double>> &fnext, vector<vector<double>> &fnow, double const& A, double om, \
		double const& t,double const& dt, \
		vector<vector<double>> &beta2, string &bc_l, string &bc_r, string &bc_b, string &bc_h, int &N)
{
	for(int i(0);i<N;++i){
      if (bc_l == "fixe"){
        fnext[i][0] = 0.0; 
	// NB: on peut aussi utiliser la condition "excitation" et poser A=0
      }else if(bc_l == "libre"){
        fnext[i][0] = fnext[i][1]; // TODO : Modifier pour imposer la condition au bord gauche libre
      }else if (bc_l =="sortie"){
        fnext[i][0] =  fnow[i][0]+sqrt(beta2[i][0])*(fnow[i][1]-fnow[i][0]); // TODO : Modifier pour imposer la condition au bord gauche "sortie de l'onde"
      }else if (bc_l == "excitation"){
        fnext[i][0] = A*sin(om*t); // TODO : Modifier pour imposer la condition au bord gauche sinusoidale
      }else{
        cerr << "Merci de choisir une condition aux bord gauche valide" << endl;
      }
	      
      if (bc_r == "fixe"){
        fnext[i][N-1] = 0.0; 
	// NB: on peut aussi utiliser la condition "excitation" et poser A=0	
      }else if(bc_r == "libre"){
        fnext[i][N-1] = fnext[i][N-2]; // TODO : Modifier pour imposer la condition au bord droit libre
      }else if (bc_r =="sortie"){
        fnext[i][N-1] = fnow[i][N-1]-sqrt(beta2[i][N-1])*(fnow[i][N-1]-fnow[i][N-2]); // TODO : Modifier pour imposer la condition au bord droit "sortie de l'onde"
      }else if (bc_r == "excitation"){ 
        fnext[i][N-1] = A*sin(om*t); // TODO : Modifier pour imposer la condition au bord droit sinusoidale
      }else{
        cerr << "Merci de choisir une condition aux bord droit valide" << endl;
      }
      
      if (bc_b == "fixe"){
        fnext[0][i] = 0.0; 
	// NB: on peut aussi utiliser la condition "excitation" et poser A=0
      }else if(bc_b == "libre"){
        fnext[0][i] = fnext[1][i]; // TODO : Modifier pour imposer la condition au bord gauche libre
      }else if (bc_b =="sortie"){
        fnext[0][i] =  fnow[0][i]+sqrt(beta2[0][i])*(fnow[1][i]-fnow[0][i]); // TODO : Modifier pour imposer la condition au bord gauche "sortie de l'onde"
      }else if (bc_b == "excitation"){
        fnext[0][i] = A*sin(om*t); // TODO : Modifier pour imposer la condition au bord gauche sinusoidale
      }else{
        cerr << "Merci de choisir une condition aux bord gauche valide" << endl;
      }
	      
      if (bc_h == "fixe"){
        fnext[N-1][i] = 0.0; 
	// NB: on peut aussi utiliser la condition "excitation" et poser A=0	
      }else if(bc_h == "libre"){
        fnext[N-1][i] = fnext[N-2][i]; // TODO : Modifier pour imposer la condition au bord droit libre
      }else if (bc_h =="sortie"){
        fnext[N-1][i] = fnow[N-1][i]-sqrt(beta2[N-1][i])*(fnow[N-1][i]-fnow[N-2][i]); // TODO : Modifier pour imposer la condition au bord droit "sortie de l'onde"
      }else if (bc_h == "excitation"){ 
        fnext[N-1][i] = A*sin(om*t); // TODO : Modifier pour imposer la condition au bord droit sinusoidale
      }else{
        cerr << "Merci de choisir une condition aux bord droit valide" << endl;
      }
  }
}

double finit(double x, double n_init, double L, double f_hat, double x1, double x2, string initialization)
{
  double finit_(0.);
  const double PI = 3.1415926535897932384626433832795028841971e0;

if(initialization=="mode"){
  // TODO: initialiser la fonction f(x,t=0) selon un mode propre
  finit_ = f_hat*cos((n_init+0.5)*x*PI/L);
}
else{
  // TODO: initialiser la fonction f(x,t=0) selon la donnée du problème
  if(x<=x1){ finit_ = 0.0;}
  else if(x>=x2){finit_ = 0.0;}
  else{finit_ = 0.5*f_hat*(1-cos(2*PI*(x-x1)/(x2-x1)));}
}
  return finit_;
}


double finit_fac(couple x, double n_init, double L, double f_hat, double x1, double x2, string initialization)
{
  double finit_(0.);
  const double PI = 3.1415926535897932384626433832795028841971e0;

if(initialization=="mode"){
  // TODO: initialiser la fonction f(x,t=0) selon un mode propre
  finit_ = f_hat*cos((n_init+0.5)*x.x1*PI/L);
}
else{
  // TODO: initialiser la fonction f(x,t=0) selon la donnée du problème
  if(x.x1<=x1){ finit_ = 0.0;}
  else if(x.x1>=x2){finit_ = 0.0;}
  else if(x.x2>=x2){finit_ = 0.0;}
  else if(x.x2<=x1){finit_ = 0.0;}
  else{finit_ = 0.5*f_hat*(1-cos(2*PI*(x.x1-x1)/(x2-x1)))*(1-cos(2*PI*(x.x2-x1)/(x2-x1)));}
}
  return finit_;
}

//
// Surcharge de l'operateur pour ecrire les elements d'un tableau
//
template <class T> ostream& operator<< (ostream& o, vector<T> const& v)
{
  unsigned int len(v.size());
  for(unsigned int i(0); i < (len - 1); ++i)
    o << v[i] << " ";
  if(len > 0)
    o << v[len-1];
  return o;
}

template <class T> ostream& operator<< (ostream& o, vector<vector<T>> const& v)
{
  unsigned int len(v.size());
  for(unsigned int i(0); i <= (len - 1); ++i){
	  for(unsigned int j(0); j <= (len - 1); ++j){
		o << v[i][j] << " ";
	}
  }
  return o;
}

ostream& operator<< (ostream& o, vector<vector<couple>> const& v)
{
  unsigned int len(v.size());
  unsigned int lenb(len);
  for(unsigned int i(0); i <= (len - 1); ++i){
	  for(unsigned int j(0); j <= (lenb - 1); ++j){
		o << v[i][j].x1 << " "<< v[i][j].x2<<" " ;
	}
  }
  return o;
}

ostream& operator<< (ostream& o, vector<couple> const& v)
{
  unsigned int len(v.size());
  for(unsigned int i(0); i <= (len - 1); ++i){
		o << v[i].x1 << " "<<v[i].x2<< " ";
  }
  return o;
}

std::vector<double> concatVectors(const std::vector<std::vector<double>>& vec_of_vecs) {
    // 1. Calculer la taille totale
    size_t total_size = 0;
    for (const auto& v : vec_of_vecs) {
        total_size += v.size();
    }

    // 2. Allouer un vecteur de la bonne taille
    std::vector<double> result;
    result.reserve(total_size);  // pour éviter les reallocations

    // 3. Copier les éléments
    for (const auto& v : vec_of_vecs) {
        result.insert(result.end(), v.begin(), v.end());
    }

    return result;
}

//
// Main
//
int main(int argc, char* argv[])
{
  const double PI = 3.1415926535897932384626433832795028841971e0;
  const double g  = 9.81;
  double dx;
  double dt;
  double t;
  double Nsteps;
  int stride(0);

  string inputPath("configuration.in"); // Fichier d'input par defaut
  if(argc>1) // Fichier d'input specifie par l'utilisateur ("./Exercice7 config_perso.in")
    inputPath = argv[1];

  ConfigFile configFile(inputPath); // Les parametres sont lus et stockes dans une "map" de strings.

  for(int i(2); i<argc; ++i) // Input complementaires ("./Exercice7 config_perso.in input_scan=[valeur]")
    configFile.process(argv[i]);

  // Parametres de simulation :
  double tfin    = configFile.get<double>("tfin");
  int nx         = configFile.get<int>("nx"); // nb intervalles
  double CFL     = configFile.get<double>("CFL");
  double nsteps  = configFile.get<double>("nsteps");
  double A       = configFile.get<double>("A");
  double f_hat   = configFile.get<double>("f_hat");
  double n_init  = configFile.get<double>("n_init");
  double hL      = configFile.get<double>("hL");
  double hR      = configFile.get<double>("hR");
  double h00     = configFile.get<double>("h00"); // profondeur, cas uniforme
  double x1      = configFile.get<double>("x1");
  double x2      = configFile.get<double>("x2");
  double xa      = configFile.get<double>("xa");
  double xb      = configFile.get<double>("xb");
  double L       = configFile.get<double>("L");
  double om      = configFile.get<double>("om");
  int n_stride(configFile.get<int>("n_stride"));
  bool fac        = configFile.get<bool>("fac");
    // Fichiers de sortie :
  string output = configFile.get<string>("output");

  ofstream fichier_x(("x_"+output).c_str());
  fichier_x.precision(15);

  ofstream fichier_v(("v_"+output).c_str());
  fichier_v.precision(15);

  ofstream fichier_f(("f_"+output).c_str());
  fichier_f.precision(15);

  ofstream fichier_en(("en_"+output).c_str());
  fichier_en.precision(15);

// Conditions aux bords:
  string bc_l           = configFile.get<string>("cb_gauche");
  string bc_r           = configFile.get<string>("cb_droite");
  string bc_b           = configFile.get<string>("cb_bas");
  string bc_h           = configFile.get<string>("cb_haut");

// Type de forme initiale de la vague: selon donnée Eq.(4) ou mode propre
// ('mode' pour mode propre, autrement Eq.(4))
  string initialization = configFile.get<string>("initialization"); 

// Onde partant vers la gauche ou vers la droite ou statique
// (par exemple 'left', 'right', 'static')
  string initial_state = configFile.get<string>("initial_state");

// Selecteur pour le cas h0 uniforme:
  bool v_uniform        = configFile.get<bool>("v_uniform");

// Selecteur pour choisir le pas de temps:
// true --> dt=tfin/nsteps; t final est exactement tfin
// false --> dt tel que beta_CFL=1; attention, t final n'est pas exactement tfin
  bool impose_nsteps    = configFile.get<bool>("impose_nsteps");
  
    bool ecrire_f = configFile.get<bool>("ecrire_f"); // Exporter f(x,t) ou non
	// Eq.(1) ou Eq.(2) [ou Eq.(6) (faculattif)]: Eq1, Eq2 ou Eq6
	string equation_type = configFile.get<string>("equation_type");
  
  int N = nx+1;                                // nb pts de maillage
if (!fac){
  vector<double> h0(N) ;
  vector<double> vel2(N) ;
  vector<double> x(N) ;
  vector<double> fpast(N), fnow(N), fnext(N), beta2(N);

  dx = L / (N-1);

  

  for(int i(0); i<N; ++i){ 
     x[i] = i * dx ;
     h0[i] = 0.0;
     if(v_uniform){
        h0[i]  = h00;
     } 
     else {
       // TODO: programmer la fonction h(x) selon la donnée
       if(x[i]<=xa and x[i]>=0){
		   h0[i]  = hL;
	   }
	   else if(x[i]<=xb){
		   h0[i]  = 0.5*(hL+hR)+0.5*(hL-hR)*cos(PI*(x[i]-xa)/(xb-xa));
	   }
	   else if(x[i]<=L){
		   h0[i]  = hR;
	   }
		   
     }
     vel2[i]  = g*h0[i];
  }
  // maiximal value of u^2 (to be used to set dt)
  auto max_vel2 = std::max_element(vel2.begin(), vel2.end());
  // TODO: set dt for given CFL
  dt = CFL*dx/sqrt(*max_vel2); // MODIFY
  // TODO: define dt and CFL with given nsteps
  if(impose_nsteps){
    dt  = tfin/nsteps; // MODIFY
    CFL = sqrt(*max_vel2)*dt/dx; // MODIFY
  }

  // Initialisation des tableaux du schema numerique :

  //TODO initialize f and beta
  for(int i(0); i<N; ++i)
  {
    fpast[i] = 0.;
    fnow[i]  = 0.;

	beta2[i] = vel2[i]*dt*dt/(dx*dx); // TODO: Modifier pour calculer beta^2 aux points de maillage

    fnow[i]  = finit(x[i], n_init,  L, f_hat, x1, x2, initialization);

    if(initial_state =="static"){
      fpast[i] = fnow[i]; // TODO: system is at rest for t<=0, 
    }
    else if(initial_state =="right"){ 
      fpast[i] = finit(x[i]+sqrt(g*h0[i])*dt, n_init,  L, f_hat, x1, x2, initialization); // TODO: propagation to the right
    }
    else if(initial_state =="left"){
      fpast[i] = finit(x[i]-sqrt(g*h0[i])*dt, n_init,  L, f_hat, x1, x2, initialization); // TODO: propagation to the left
    }
  }


  cout<<"beta2[0] is "<<beta2[0]<<endl;
  cout<<"dt is "<< dt <<endl;
  cout<<"dx is "<<dx <<endl;
  cout<<"dx/dt is "<<dx/dt<<endl;


  // Boucle temporelle :
  for(t=0.; t<tfin-.5*dt; t+=dt)
  {
    // Ecriture :
    if(stride%n_stride == 0)
    {
      if(ecrire_f) fichier_f << t << " " << fnow << endl;
      fichier_en << t << " " << energy(fnow,dx) << endl;
     }
    ++stride;

    // Evolution :
    for(int i(1); i<N-1; ++i)
    {
      fnext[i] = 0.0; // TODO : Schémas pour les 3 cas, Equation A ou B ou C
      
     if (equation_type == "A") {
		fnext[i] = 2.0 * (1-beta2[i]) *fnow[i] - fpast[i] + beta2[i] *(fnow[i+1] + fnow[i-1]);
	}
	else if (equation_type == "B") {
		fnext[i] = 2.0 * (1-beta2[i])  *fnow[i] - fpast[i] + beta2[i]*(fnow[i+1] + fnow[i-1]) + 0.25* (beta2[i+1]-beta2[i-1])*(fnow[i+1] - fnow[i-1]);
	}
	else if (equation_type == "C") {
		fnext[i] = - fpast[i] + (2.0 + beta2[i+1]- 4*beta2[i] +beta2[i-1])* fnow[i] + (fnow[i+1] + fnow[i-1]) * beta2[i] + 0.5*(beta2[i+1]-beta2[i-1])*(fnow[i+1] - fnow[i-1]);
	}
    }

    // Impose boundary conditions
    boundary_condition(fnext, fnow, A, om, t, dt, beta2, bc_l, bc_r, N);

    // Mise a jour et préparer le pas suivant:
    fpast = fnow;
    fnow  = fnext;
  }

  if(ecrire_f) fichier_f << t << " " << fnow << endl;
  fichier_x << x << endl;
  fichier_v << vel2 << endl;
  fichier_en << t << " " << energy(fnow,dx) << endl;

  fichier_f.close();
  fichier_x.close();
  fichier_v.close();
  fichier_en.close();
}else{
	
  vector<vector<double>> h0(N,vector<double>(N)) ;
  vector<vector<double>> vel2(N,vector<double>(N)) ;
  vector<vector<couple>> x(N,vector<couple>(N));
  vector<vector<double>> fpast(N,vector<double>(N)), fnow(N,vector<double>(N)), fnext(N,vector<double>(N)), beta2(N,vector<double>(N));
  
  dx = L / (N-1);
  
  for(int i(0); i<N; ++i){ 
	  for(int j(0); j<N; ++j){ 
     x[i][j] = {i * dx,j*dx} ;
     h0[i][j] = 0.0;
     if(v_uniform){
        h0[i][j]  = h00;
     } 
     else {
       // TODO: programmer la fonction h(x) selon la donnée
       h0[i][j]= L*cos((x[i][j].x1 - 0.5*L)*(x[i][j].x2 - 0.5*L))*cos((x[i][j].x1 - 0.5*L)*(x[i][j].x2 - 0.5*L));
		   
     }
     vel2[i][j]  = g*h0[i][j];
  }
}
  
  // maiximal value of u^2 (to be used to set dt)
  auto max_vel2 = std::max_element(concatVectors(vel2).begin(), concatVectors(vel2).end());
  // TODO: set dt for given CFL
  dt = CFL*dx/(*max_vel2); // MODIFY
  // TODO: define dt and CFL with given nsteps
  if(impose_nsteps){
    dt  = tfin/nsteps; // MODIFY
    CFL = sqrt(*max_vel2)*dt/dx; // MODIFY
  }

  // Initialisation des tableaux du schema numerique :

  //TODO initialize f and beta
  for(int i(0); i<N; ++i)
  {
	  for(int j(0); j<N; ++j){
    fpast[i][j] = 0.;
    fnow[i][j]  = 0.;
	beta2[i][j] = vel2[i][j]*dt*dt/(dx*dx); // TODO: Modifier pour calculer beta^2 aux points de maillage
    fnow[i][j]  = finit_fac(x[i][j], n_init,  L, f_hat, x1, x2, initialization);

    if(initial_state =="static"){
      fpast[i][j] = fnow[i][j]; // TODO: system is at rest for t<=0, 
    }
    else if(initial_state =="right"){
		couple k=x[i][j];
		k.x1 += sqrt(g*h0[i][j])*dt;
      fpast[i][j] = finit_fac(k, n_init,  L, f_hat, x1, x2, initialization); // TODO: propagation to the right
    }
    else if(initial_state =="left"){
		couple k=x[i][j];
		k.x1 -= sqrt(g*h0[i][j])*dt;
      fpast[i][j] = finit_fac(k, n_init,  L, f_hat, x1, x2, initialization); // TODO: propagation to the left
    }
  }
}


  cout<<"beta2[0][0] is "<<beta2[0][0]<<endl;
  cout<<"dt is "<< dt <<endl;
  cout<<"dx is "<<dx <<endl;
  cout<<"dx/dt is "<<dx/dt<<endl;


  // Boucle temporelle :
  for(t=0.; t<tfin-.5*dt; t+=dt)
  {
    // Ecriture :
    if(stride%n_stride == 0)
    {
      if(ecrire_f) fichier_f << t << " " << fnow << endl;
      fichier_en << t << " " << energy_fac(fnow,dx) << endl;
     }
    ++stride;

    // Evolution :
    for(int i(1); i<N-1; ++i)
    {
		for(int j(1); j<N-1; ++j)
    {
      fnext[i][j] = 0.0; // TODO : Schémas pour les 3 cas, Equation A ou B ou C
      
     if (equation_type == "A") {
    fnext[i][j] = 2.0 * (1 - 2*beta2[i][j]) * fnow[i][j] - fpast[i][j] + 
                  beta2[i][j] * (fnow[i + 1][j] + fnow[i - 1][j] + 
                  fnow[i][j + 1] + fnow[i][j - 1]);
}
	else if (equation_type == "B") {
    fnext[i][j] = 2.0 * (1 - 2*beta2[i][j]) * fnow[i][j] - fpast[i][j] + 
                  (beta2[i][j] + 0.25 * (beta2[i + 1][j] - beta2[i - 1][j])) * 
                  (fnow[i + 1][j] + fnow[i - 1][j]) + 
                  (beta2[i][j] + 0.25 * (beta2[i][j + 1] - beta2[i][j - 1])) * 
                  (fnow[i][j + 1] + fnow[i][j - 1]);
}
	else if (equation_type == "C") {
    fnext[i][j] = -fpast[i][j] + 
                  ((2.0 + beta2[i + 1][j] - 4.0 * beta2[i][j] + beta2[i - 1][j]) + 
                   (2.0 + beta2[i][j + 1] - 4.0 * beta2[i][j] + beta2[i][j - 1])) * 
                   fnow[i][j] + 
                  (fnow[i + 1][j] + fnow[i - 1][j]) * 
                  (beta2[i][j] + 0.5 * (beta2[i + 1][j] - beta2[i - 1][j])) + 
                  (fnow[i][j + 1] + fnow[i][j - 1]) * 
                  (beta2[i][j] + 0.5 * (beta2[i][j + 1] - beta2[i][j - 1]));
}
    }
}

    // Impose boundary conditions
    boundary_condition_fac(fnext, fnow, A, om, t, dt, beta2, bc_l, bc_r,bc_b,bc_h, N);

    // Mise a jour et préparer le pas suivant:
    fpast = fnow;
    fnow  = fnext;
  }

  if(ecrire_f) fichier_f << t << " " << fnow << endl;
  fichier_x << x << endl;
  fichier_v << vel2 << endl;
  fichier_en << t << " " << energy_fac(fnow,dx) << endl;

  fichier_f.close();
  fichier_x.close();
  fichier_v.close();
  fichier_en.close();
  
	
}
  return 0;
}




