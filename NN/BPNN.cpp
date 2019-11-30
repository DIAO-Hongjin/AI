#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream> 
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace std;

int dimension;
int data_num;
vector< string > attribute;
map< string, vector< double > > all_data;
map< string, double > max_value;
map< string, double > min_value;
vector< string > dteday;
vector< int > realy_count;
vector< int > train_id;
vector< int > validation_id;

int layers;
double n;
vector< int > nodes;
map< int, vector< vector< double > > > weights;
map< int, vector< vector< double > > > weight_step;
map< int, vector< double > > bias;
map< int, vector< double > > bias_step;

template< class out_type, class in_type >  
out_type convert( const in_type & value ) {
    stringstream stream;  
    out_type result;
    
    stream << value;  
    stream >> result; 
    
    return result;
}

vector< double > NormalizeVector( vector< double > vec, string attr ) {
	double max = 0, min = 1000;
	vector< double > result;
	
	for( int i = 0; i < vec.size(); i ++ ) {  //�ҵ����ֵ����Сֵ 
		if( max < vec[ i ] )
		    max = vec[ i ];
		if( min > vec[ i ] )
		    min = vec[ i ];
	}
	max_value[ attr ] = max;  //��¼�������������ֵ����Сֵ 
	min_value[ attr ] = min;
	for( int i = 0; i < vec.size(); i ++ )  //��һ�� 
		result.push_back( ( vec[ i ] - min ) / ( max - min ) );
	
	return result;
}

//���Լ���ѵ���������ֵ����Сֵ���й�һ�� 
double NormalizeValue( double val, string attr ) {
	return ( val - min_value[ attr ] ) / ( max_value[ attr ] - min_value[ attr ] );
}

void PartitionDataSet() {
	for( int i = 0; i < data_num; i ++ ) {
		if( i % 10 == 0 )  //����1������11����...����10n+1����������Ϊ��֤�� 
			validation_id.push_back( i );
		else  //ʣ����������Ϊѵ���� 
		    train_id.push_back( i );
	}
}

void ReadData() {
	bool flag = true;
	int temp_col = 0;
	string line = "", str = "";
	ifstream in( "F://AI/Lab6NN/BPNN_Dataset/train.csv" ); 
	
	while( getline( in, line ) ) {
		temp_col = 0;
		if( flag ) {
			for( int i = 0; i <= line.size(); i ++ ) {
				if( line[ i ] == ',' ) {
					if( temp_col == 2 ) {
						dimension ++;
						attribute.push_back( str );
					}
					else if( temp_col > 3 ) {
						dimension ++;
						attribute.push_back( str );
					}
					str = "";
					temp_col ++;
				}
				else if( i == line.size() )
					str = "";
				else
					str += line[ i ];
		    }
			flag = false;
		}
		else {
			data_num ++;
			for( int i = 0; i <= line.size(); i ++ ) {
				if( line[ i ] == ',' ) {
					if( temp_col == 1 ) 
						dteday.push_back( str );
					else if( temp_col == 2 ) 
						all_data[ attribute[ temp_col - 2 ] ].push_back( convert< double, string >( str ) );
					else if( temp_col > 3 ) 
					    all_data[ attribute[ temp_col - 3 ] ].push_back( convert< double, string >( str ) );
					str = "";
					temp_col ++;
				}
				else if( i == line.size() ) {
					realy_count.push_back( convert< int, string >( str ) );
					str = "";
				}
				else
					str += line[ i ];
			}
		}
	}
	/*for( map< string, vector< double > > :: iterator it = all_data.begin(); it != all_data.end(); it ++ ) 
		if( it -> first != "holiday" && it -> first != "workingday" )
		    it -> second = NormalizeVector( it -> second, it -> first );*/ 
	PartitionDataSet();
	
	in.close();
} 

vector< vector< double > > GetTrainData() {
	vector< double > temp;
	vector< vector< double > > train_data;
	
	train_data.clear();
	for( int i = 0; i < train_id.size(); i ++ ) {
		temp.clear();
		for( int j = 0; j < attribute.size(); j ++ ) 
		    temp.push_back( all_data[ attribute[ j ] ][ train_id[ i ] ] );
		train_data.push_back( temp );
	}
	
	return train_data;
}

vector< vector< double > > GetValidationData() {
	vector< double > temp;
	vector< vector< double > > validation_data;
	
	validation_data.clear();
	for( int i = 0; i < validation_id.size(); i ++ ) {
		temp.clear();
		for( int j = 0; j < attribute.size(); j ++ ) 
		    temp.push_back( all_data[ attribute[ j ] ][ validation_id[ i ] ] );
		validation_data.push_back( temp );
	}
	
	return validation_data;
}

vector< vector< double > > GetAllData() {
	vector< double > temp;
	vector< vector< double > > data;
	
	data.clear();
	for( int i = 0; i < data_num; i ++ ) {
		temp.clear();
		for( int j = 0; j < attribute.size(); j ++ ) 
		    temp.push_back( all_data[ attribute[ j ] ][ i ] );
		data.push_back( temp );
	}
	
	return data;
}

double GetDotProduct( vector< double > vec1, vector< double > vec2, int length ) {
	double result = 0;;
	
	for( int i = 0; i < length; i ++ )
		result += vec1[ i ] * vec2[ i ];
	
	return result; 
}

//������룺Input=wx+�� 
double GetInput( vector< double > cur_weight, vector< double > cur_data, double cur_bias, int length ) {
	return GetDotProduct( cur_weight, cur_data, length ) + cur_bias;
}

//�������sigmoid������f(x)=1/(1+e^-x) 
double Sigmoid( double x ) {
	return 1.0 / ( 1.0 + exp( -1.0 * x ) );
}

//sigmoid�����ĵ�����f'(x)=f(x)(1-f(x)) 
double GetDerSigmoid( double x ) {
	return Sigmoid( x ) * ( 1 - Sigmoid( x ) );
}

void Inatial() {  //��ʼ�����в��� 
	vector< double > temp_weight;
	vector< double > temp_step;
	
	n = 0.00001;  //ѧϰ�� 
	layers = 4;  //������4�������1+���ز�2+�����1�� 
	srand( ( unsigned )time( NULL ) );
	for( int i = 0; i < layers; i ++ ) {  //��ʼ��ÿ��������� 
		if( i == 0 )  //����㣺�������Ϊ�������� 
		    nodes.push_back( attribute.size() );
		else if( i == layers - 1 )  //����㣺�������Ϊ1 
		    nodes.push_back( 1 );
	    else  //���ز㣺���е��� 
	        nodes.push_back( 6 );
	}
	for( int i = 0; i < layers - 1; i ++ ) {  
		for( int j = 0; j < nodes[ i + 1 ]; j ++ ) {
			temp_weight.clear();
			temp_step.clear();
			for( int k = 0; k < nodes[ i ]; k ++ ) {
				temp_weight.push_back( ( rand() % 1000 ) * 1.0 / 1001 );
				temp_step.push_back( 0 );
			}
			weights[ i ].push_back( temp_weight );  //Ȩ�أ���� 
			weight_step[ i ].push_back( temp_step );  //Ȩ�ز�����0 
			bias[ i ].push_back( ( rand() % 1000 ) * 1.0 / 1001 );  //ƫ�ã���� 
			bias_step[ i ].push_back( 0 );  //ƫ�ò�����0 
		}
	}
}

//������ 
vector< double > ForwordPass( vector< double > cur_input, int to_layer ) {
    vector< double > output;
    
    output.clear();
	if( to_layer == layers - 1 )  //����㣺���=���� 
		output.push_back( cur_input[ 0 ] );
	else  //���ز㣺���=sigmoid�����룩 
		for( int i = 0; i < nodes[ to_layer ]; i ++ )
			output.push_back( Sigmoid( cur_input[ i ] ) );
	
	return output;
}

//�����ݶ���� 
vector< vector< double > > GetErrorGrads( vector< vector< double > > input_data, vector< vector< double > > output_data, int data_id ) {
	double temp_val;
	vector< double > temp_vec;
	vector< vector< double > > error_grads;
	
	error_grads.clear();
	for( int i = layers - 1, cnt = 0; i > 0; i --, cnt ++ ) {
		temp_vec.clear();
		if( i == layers - 1 )  //����㣺err=T-O 
			temp_vec.push_back( realy_count[ data_id ] * 1.0 - output_data[ i ][ 0 ] );
		else {  //���ز㣺err=f'(I)��err w 
			for( int j = 0; j < nodes[ i ]; j ++ ) {
				temp_val = 0;
				for( int k = 0; k < nodes[ i + 1 ]; k ++ )
			        temp_val += error_grads[ cnt - 1 ][ k ] * weights[ i ][ k ][ j ];
			    temp_vec.push_back( temp_val * GetDerSigmoid( input_data[ i ][ j ] ) );
			}
		}
		error_grads.push_back( temp_vec );
	}
	
	return error_grads;
}

//���²��� 
void RenewStep( vector< vector< double > > output_data, vector< vector< double > > error_grads ) {
	for( int i = layers - 2, cnt = 0; i >= 0; i --, cnt ++ ) {
		for( int j = 0; j < nodes[ i + 1 ]; j ++ ) {
			for( int k = 0; k < nodes[ i ]; k ++ ) //��w = err O 
				weight_step[ i ][ j ][ k ] = error_grads[ cnt ][ j ] * output_data[ i ][ k ];
			bias_step[ i ][ j ] = error_grads[ cnt ][ j ]; //����=err 
		}
	} 
}

//����Ȩ�غ�ƫ�� 
void RenewWeightAndBias() {
	for( int i = 0; i < layers - 1; i ++ ) {	
		for( int j = 0; j < nodes[ i + 1 ]; j ++ ) {
			for( int k = 0; k < nodes[ i ]; k ++ )  //w = w + �ǡ�w 
				weights[ i ][ j ][ k ] += n * weight_step[ i ][ j ][ k ];
			bias[ i ][ j ] += n * bias_step[ i ][ j ];  //��= ��+ �ǡ��� 
		}
	}
}

//���򴫵ݸ���Ȩ�� 
void BackwardPass( vector< vector< double > > input_data, 
                   vector< vector< double > > output_data, int data_id ) {
	//�����ݶ������²���
	RenewStep( output_data, GetErrorGrads( input_data, output_data, data_id ) );   
	//����Ȩ��
	RenewWeightAndBias();   
}

double GetValidateLoss() {
	double temp_lose = 0;
	vector< vector< double > > validation_data = GetValidationData();
	vector< vector< double > > input_data;
	vector< vector< double > > output_data;
	vector< double > temp_vec;
	
	for( int i = 0; i < validation_id.size(); i ++ ) {
		input_data.clear();
       	output_data.clear();
		input_data.push_back( validation_data[ i ] );
		output_data.push_back( validation_data[ i ] );
		for( int j = 1; j < layers; j ++ ) {
			temp_vec.clear();
			for( int k = 0; k < nodes[ j ]; k ++ )
				temp_vec.push_back( GetInput( weights[ j - 1 ][ k ], output_data[ j - 1 ], bias[ j - 1 ][ k ], nodes[ j - 1 ] ) );
			input_data.push_back( temp_vec );
			output_data.push_back( ForwordPass( input_data[ j ], j ) );
		}
		temp_lose += pow( ( ( int )output_data[ layers - 1 ][ 0 ] - realy_count[ validation_id[ i ] ] ) * 1.0, 2 );
	}
	return temp_lose * 1.0 / validation_id.size();
}

void BPNN() {
	int iterator = 5000;  //�������� 
	double temp_lose1 = 0, temp_lose2 = 0;
	vector< vector< double > > train_data = GetTrainData();  //�õ�ѵ�������� 
	vector< vector< double > > input_data;  //ÿ���������
	vector< vector< double > > output_data;  //ÿ�������� 
	vector< double > temp_vec;
	ofstream out1( "train_loss.txt" );
	ofstream out2( "val_loss.txt" );
	
	Inatial();  //��ʼ��������� 
	for( int i = 0; i < iterator; i ++ ) {  //������ָ������ 
		temp_lose1 = 0;
		temp_lose2 = 0;
		for( int j = 0; j < train_id.size(); j ++ ) {  //ÿ�ζ�ȡһ��ѵ��������
			input_data.clear();
        	output_data.clear();
			input_data.push_back( train_data[ j ] );  //����㣺����Ϊѵ������
			output_data.push_back( train_data[ j ] ); //�����������ͬ 
			for( int k = 1; k < layers; k ++ ) {  //ǰ�򴫵ݣ��ֱ�������ز����������������
				temp_vec.clear();
				for( int l = 0; l < nodes[ k ]; l ++ )
				    temp_vec.push_back( GetInput( weights[ k - 1 ][ l ], output_data[ k - 1 ], bias[ k - 1 ][ l ], nodes[ k - 1 ] ) );
				input_data.push_back( temp_vec );  //������� 
				output_data.push_back( ForwordPass( input_data[ k ], k ) );  //������ 
			}
			temp_lose1 += pow( ( ( int )output_data[ layers - 1 ][ 0 ] - realy_count[ train_id[ j ] ] ) * 1.0, 2 );
			BackwardPass( input_data, output_data, train_id[ j ] );    //���򴫵ݣ�����Ȩ�� 
		}
		temp_lose2 = GetValidateLoss();
		out1 << temp_lose1 * 1.0 / train_id.size() << endl;
		out2 << temp_lose2 << endl;
	}
	out1.close();
	out2.close();
}

void Validate() {
	vector< vector< double > > data = GetAllData();
	vector< vector< double > > input_data;
	vector< vector< double > > output_data;
	vector< double > temp_vec;
	ofstream out( "result.txt" );
	
	for( int i = 0; i < data_num; i ++ ) {
		input_data.clear();
       	output_data.clear();
		input_data.push_back( data[ i ] );
		output_data.push_back( data[ i ] );
		for( int j = 1; j < layers; j ++ ) {
			temp_vec.clear();
			for( int k = 0; k < nodes[ j ]; k ++ )
				temp_vec.push_back( GetInput( weights[ j - 1 ][ k ], output_data[ j - 1 ], bias[ j - 1 ][ k ], nodes[ j - 1 ] ) );
			input_data.push_back( temp_vec );
			output_data.push_back( ForwordPass( input_data[ j ], j ) );
		}
		out << dteday[ i ] << " " 
		    << ( int )output_data[ layers - 1 ][ 0 ] << " " << realy_count[ i ] << endl;
	}
	out.close();
}

void Test() {
	bool flag = true;
	int temp_col = 0;
	int test_num = 0;
	string line = "", str = "";
	vector< double > temp;
	map< string, vector< double > > test_data;
	ifstream in( "F://AI/Lab6NN/BPNN_Dataset/test.csv" ); 
	ofstream out( "15352076_diaohongjin.txt" );
	
	test_data.clear();
	while( getline( in, line ) ) {
		temp_col = 0;
		if( flag ) 
			flag = false;
		else {
			test_num ++;
			for( int i = 0; i <= line.size(); i ++ ) {
				if( line[ i ] == ',' ) {
					if( temp_col == 2 ) 
						test_data[ attribute[ temp_col - 2 ] ].push_back( convert< double, string >( str ) );
					else if( temp_col > 3 ) 
					    test_data[ attribute[ temp_col - 3 ] ].push_back( convert< double, string >( str ) );
					str = "";
					temp_col ++;
				}
				else if( i == line.size() ) 
					str = "";
				else
					str += line[ i ];
			}
		}
	}
	for( map< string, vector< double > > :: iterator it = test_data.begin(); it != test_data.end(); it ++ ) {
		if( it -> first != "holiday" && it -> first != "workingday" ) {
			vector< double > temp2 = it -> second;
			for( int i = 0; i < temp2.size(); i ++ )
			    temp2[ i ] = NormalizeValue( temp2[ i ], it -> first );
			it -> second = temp2;
		}
	}
	vector< double > temp3, temp_vec;
	vector< vector< double > > input_data;
	vector< vector< double > > output_data;
	for( int i = 0; i < test_num; i ++ ) {
		temp3.clear();
		for( int j = 0; j < attribute.size(); j ++ ) 
		    temp3.push_back( test_data[ attribute[ j ] ][ i ] );
		input_data.clear();
       	output_data.clear();
		input_data.push_back( temp3 );
		output_data.push_back( temp3 );
		for( int j = 1; j < layers; j ++ ) {
			temp_vec.clear();
			for( int k = 0; k < nodes[ j ]; k ++ )
				temp_vec.push_back( GetInput( weights[ j - 1 ][ k ], output_data[ j - 1 ], bias[ j - 1 ][ k ], nodes[ j - 1 ] ) );
			input_data.push_back( temp_vec );
			output_data.push_back( ForwordPass( input_data[ j ], j ) );
		}
		out << ( int )output_data[ layers - 1 ][ 0 ] << endl;
	}
	
	in.close();
	out.close();
}

int main() {
	ReadData();
	BPNN();
	Validate();
	//Test();
	
	return 0;
}
