#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream> 
#include <cmath>

using namespace std;

int dimension;
int train_vec_num;
vector< vector< double > > train_vector;
vector< int > train_label;
vector< double > weight;

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
	ifstream in( "F://AI/lab3(PLA)/lab3����/train.csv" ); 
	
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
				train_label.push_back( convert< int, string >( num ) );
				num = "";
			}
			else
				num += line[ i ];
		}
		train_vector.push_back( temp ); 
		train_vec_num ++;
		flag = false;
	}
	
	in.close();
} 

double GetDotProduct( vector< double > vec1, vector< double > vec2, int length )
{
	double result = 0;;
	
	for( int i = 0; i < length; i ++ )
		result += vec1[ i ] * vec2[ i ];
	
	return result; 
}

int SignFunction( double num )
{
	if( num > 0 )
	    return 1;
	if( num < 0 )
	    return -1;
	return 0;
}

vector< double > RenewWeight( vector< double > old_weight, vector< double > x, int y, int length )
{
	vector< double > result;
	
	result.clear();
	for( int i = 0; i < length; i ++ )
	    result.push_back( old_weight[ i ] + y * 1.0 * x[ i ] );
	
	return result;
}

void GetWeight()  //ѵ��ģ�ͣ��õ�Ȩ�� 
{
	int cnt = 0, iterator = 10000;  //��ǰѭ��������ָ���������� 
	weight.clear();
	for( int i = 0; i < dimension; i ++ )  //��ʼ��ȨֵΪ0 
	    weight.push_back( 0 );
	for( int i = 0; i < train_vec_num; i ++ )  //��������ѵ�������� 
	{
		//Ԥ��������ǩΪsign( w��x )���ж��Ƿ�����ȷ��ǩ���
		if( SignFunction( GetDotProduct( weight, train_vector[ i ], dimension ) ) != train_label[ i ] )
		{
			//����ȷ��ǩ�����������Ȩ��Ϊ w+yx
			weight = RenewWeight( weight, train_vector[ i ], train_label[ i ], dimension );
			//��Ϊѵ�����������Կɷֵģ�Ϊ�����ó���ͣ����������һ����������
			//���û�дﵽ���������������Ȩ�غ��ͷ��ʼ����ѵ����
			if( cnt < iterator )   
			{
				i = -1;
				cnt ++;
			}
		}
	}
}

int InitialPLA( vector< double > vec )  //����Ȩ��Ԥ���ǩ 
{
	//����Ԥ��ı�ǩ��sign( w��x )
	return SignFunction( GetDotProduct( weight, vec, dimension ) );
}

void Validate()
{
	int validation_vec_num = 0; 
	string line = "", num = "";
	int tp = 0, fn = 0, tn = 0, fp = 0;
	double accuracy = 0, recall = 0, precision = 0, f1 = 0;
	vector< double > temp;
	vector< int > validation_label;
	vector< int > predict_validation_label;
	vector< vector< double > > validation_vector;
	ifstream in( "F://AI/lab3(PLA)/lab3����/val.csv" ); 
	
	validation_label.clear();
	predict_validation_label.clear();
	validation_vector.clear(); 
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
			{
				validation_label.push_back( convert< int, string >( num ) );
				num = "";
			}
			else
				num += line[ i ];
		}
		validation_vector.push_back( temp );
		validation_vec_num ++;
	}
	for( int i = 0; i < validation_vec_num; i ++ )
		predict_validation_label.push_back( InitialPLA( validation_vector[ i ] ) );
	for( int i = 0; i < validation_vec_num; i ++ )
	{
		if( validation_label[ i ] == 1 && predict_validation_label[ i ] == 1 )
		    tp ++;
		if( validation_label[ i ] == 1 && predict_validation_label[ i ] == -1 )
		    fn ++;
		if( validation_label[ i ] == -1 && predict_validation_label[ i ] == -1 )
		    tn ++;
		if( validation_label[ i ] == -1 && predict_validation_label[ i ] == 1 )
		    fp ++;
	}
	accuracy = ( tp + tn ) * 1.0 / ( tp + fn + tn + fp );
	recall = tp * 1.0 / ( tp + fn );
	precision = tp * 1.0 / ( tp + fp );
	f1 = 2.0 * precision * recall / ( recall + precision );
	cout << "��֤��������" << validation_vec_num << endl;
	cout << "tp��" << tp << endl << "fn��" << fn << endl
	     << "tn��" << tn << endl << "fp��" << fp << endl; 
	cout << "׼ȷ�ʣ�" << accuracy << endl << "��ȷ�ʣ�" << precision << endl
		 << "�ٻ��ʣ�" << recall << endl << "Fֵ��" << f1 << endl;
		 
	in.close();
}

void Test()
{
	int test_vec_num = 0;
	string line = "", num = "";
	vector< double > temp;
	vector< int > predict_test_label;
	vector< vector< double > > test_vector;
	ifstream in( "F://AI/lab3(PLA)/lab3����/test.csv" );  
	ofstream out( "15352076_diaohongjin_PLA.csv" );
	
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
		predict_test_label.push_back( InitialPLA( test_vector[ i ] ) );
	for( int i = 0; i < test_vec_num; i ++ )
	    out << predict_test_label[ i ] << endl;
	
	in.close();
	out.close();
}

int main()
{
	ReadTrainData();
	GetWeight();
	Validate();
	Test();
	
	return 0;
}
