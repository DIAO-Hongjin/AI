#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream> 
#include <cmath>

using namespace std;

const double e = 2.718281828459;

int dimension;
int all_vec_num, train_vec_num, val_vec_num;
vector< vector< double > > all_vector, train_vector, val_vector;
vector< int > all_label, train_label, val_label;
vector< double > weight;

void PrintVector( vector< double > vec ) 
{
	for( int i = 0; i < vec.size(); i ++ )
	    cout << vec[ i ] << " ";
	cout << endl;
}

template< class out_type, class in_type >  
out_type convert( const in_type & value ) 
{
    stringstream stream;  
    out_type result;
    
    stream << value;  
    stream >> result; 
    
    return result;
}

void ReadTrainData()
{
	bool flag = true;
	string line = "", num = "";
	vector< double > temp;
	ifstream in( "F://AI/lab5_LR/train.csv" ); 
	
	dimension ++;
	while( getline( in, line ) ) 
	{
		temp.clear();
		temp.push_back( 1 );
		for( int i = 0; i <= line.size(); i ++ )
		{
			if( line[ i ] == ',' )
			{
				temp.push_back( convert< double, string >( num ) );
				num = "";
				if( flag )
					dimension ++;
			}
			else if( i == line.size() )
			{
				all_label.push_back( convert< int, string >( num ) );
				num = "";
			}
			else
				num += line[ i ];
		}
		all_vector.push_back( temp ); 
		all_vec_num ++;
		flag = false;
	}
	
	in.close();
} 

void PartitionDataSet()
{
	for( int i = 0; i < all_vec_num; i ++ )
	{
		if( i % 10 == 0 )
		{
			val_vector.push_back( all_vector[ i ] );
			val_label.push_back( all_label[ i ] );
			val_vec_num ++;
		}
		else
		{
			train_vector.push_back( all_vector[ i ] );
			train_label.push_back( all_label[ i ] );
			train_vec_num ++;
		}
	}
}

double GetDotProduct( vector< double > vec1, vector< double > vec2, int length )
{
	double result = 0;;
	
	for( int i = 0; i < length; i ++ )
		result += vec1[ i ] * vec2[ i ];
	
	return result; 
}

double LogisticFunction( double num )
{
	return pow( e, num ) / ( 1 + pow( e, num ) );
} 

int GetLabel( double num )
{
	if( num > 0.5 )
	    return 1;

	return 0;
}

double GetVectorLength( vector< double > vec )
{
	double length = 0;
	
	for( int i = 0; i < vec.size(); i ++ )
	    length += pow( vec[ i ], 2 );
	
	return sqrt( length );
}

void GetWeight( double n ) 
{
	int iterator = 500;
	double tempwi = 0, temp_cost = 0;
	vector< double > temp_weight;
	vector< double > temp_cost_vector; 
	
	weight.clear();
	for( int i = 0; i < dimension; i ++ )
		weight.push_back( 1 );
	for( int i = 0; i < iterator; i ++ ) 
	{
		temp_weight.clear();
		temp_cost_vector.clear();
		for( int j = 0; j < dimension; j ++ )
		{
			temp_cost = 0;
			for( int k = 0; k < train_vec_num; k ++ )
				temp_cost += ( LogisticFunction( GetDotProduct( weight, train_vector[ k ], dimension ) ) 
				             - train_label[ k ] * 1.0 ) * train_vector[ k ][ j ];
			temp_cost_vector.push_back( temp_cost );
			temp_weight.push_back( weight[ j ] - n * temp_cost );
		}
		weight = temp_weight;
		if( GetVectorLength( temp_cost_vector ) == 0 )
		    break;
	}
}

int LR( vector< double > vec )
{
	return GetLabel( LogisticFunction( GetDotProduct( weight, vec, dimension ) ) );
}

void Validate()
{
	int cnt = 0;
	for( int i = 0; i < val_vec_num; i ++ )
	{
		if( val_label[ i ] == LR( val_vector[ i ] ) )
		    cnt ++;
	}
	cout << "×¼È·ÂÊ£º" << cnt * 1.0 / val_vec_num << endl;
}

void Test()
{
	int test_vec_num = 0;
	string line = "", num = "";
	vector< double > temp;
	vector< int > predict_test_label;
	vector< vector< double > > test_vector;
	ifstream in( "F://AI/lab5_LR/test.csv" );  
	ofstream out( "15352076_diaohongjin.csv" );
	
	predict_test_label.clear();
	test_vector.clear();
	while( getline( in, line ) ) 
	{
		temp.clear();
		temp.push_back( 1 );
		for( int i = 0; i <= line.size(); i ++ )
		{
			if( line[ i ] == ',' )
			{
				temp.push_back( convert< double, string >( num ) );
				num = "";
			}
			else if( i == line.size() )
				num = "";
			else
				num += line[ i ];
		}
		test_vector.push_back( temp );
		test_vec_num ++;
	}
	for( int i = 0; i < test_vec_num; i ++ )
		predict_test_label.push_back( LR( test_vector[ i ] ) );
	for( int i = 0; i < test_vec_num; i ++ )
	    out << predict_test_label[ i ] << endl;
	
	in.close();
	out.close();
}

int main()
{
	ReadTrainData();
	PartitionDataSet();
	GetWeight( 0.000009 );
	Validate();
	
	//Test();
	
	return 0;
}
