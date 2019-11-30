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
vector< double > best_weight; 

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
	ifstream in( "F://AI/lab3(PLA)/lab3数据/train.csv" ); 
	
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
	return -1;
}

vector< double > RenewWeight( vector< double > old_weight, vector< double > x, int y, int length )
{
	vector< double > result;
	
	result.clear();
	for( int i = 0; i < length; i ++ )
	    result.push_back( old_weight[ i ] + y * 1.0 * x[ i ] );
	
	return result;
}

double GetErrorRate( vector< double > weight, vector< vector< double > > sample, vector< int > label, int vec_num, int vec_length )
{
	int cnt = 0;
	
	for( int i = 0; i < vec_num; i ++ )
		if( SignFunction( GetDotProduct( weight, sample[ i ], vec_length ) ) != label[ i ] )
		    cnt ++;
	
	return cnt * 1.0 / vec_num;
}

void GetWeight()  //训练模型，得到权重
{
	int cnt = 0, iterator = 25;  //当前循环次数和指定迭代次数 
	
	weight.clear();
	best_weight.clear();
	for( int i = 0; i < dimension; i ++ )  //初始化权重为0 
	{
		weight.push_back( 0 );
		best_weight.push_back( 0 );
	}
	for( int i = 0; i < train_vec_num; i ++ ) //遍历所有训练集样本
	{
		if( SignFunction( GetDotProduct( weight, train_vector[ i ], dimension ) ) != train_label[ i ] )  //预测样本标签为sign( w·x )，判断是否与正确标签相符 
		{
			weight = RenewWeight( weight, train_vector[ i ], train_label[ i ], dimension );  //与正确标签不相符，更新权重为 w+yx 
			if( GetErrorRate( weight, train_vector, train_label, train_vec_num, dimension )  //判断当前权重的错误率大小 
			    < GetErrorRate( best_weight, train_vector, train_label, train_vec_num, dimension ) )
				best_weight = weight;  //如果当前权重的预测错误率更低，则将最优权重更新为当前权重 
			if( cnt < iterator )  //如果没有达到指定迭代次数，则更新权重后从头开始遍历训练集  
			{
				i = -1;
				cnt ++;
			}
		}
	}
}

int PocketPLA( vector< double > vec )  //利用权重预测标签 
{
	return SignFunction( GetDotProduct( best_weight, vec, dimension ) );  //返回预测的标签：sign( w·x )
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
	ifstream in( "F://AI/lab3(PLA)/lab3数据/val.csv" ); 
	
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
		predict_validation_label.push_back( PocketPLA( validation_vector[ i ] ) );
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
	cout << "验证集总数：" << validation_vec_num << endl;
	cout << "tp：" << tp << endl << "fn：" << fn << endl
	     << "tn：" << tn << endl << "fp：" << fp << endl; 
	cout << "准确率：" << accuracy << endl << "精确率：" << precision << endl
		 << "召回率：" << recall << endl << "F值：" << f1 << endl;
	
	in.close();
}

void Test()
{
	int test_vec_num = 0;
	string line = "", num = "";
	vector< double > temp;
	vector< int > predict_test_label;
	vector< vector< double > > test_vector;
	ifstream in( "F://AI/lab3(PLA)/lab3数据/test.csv" ); 
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
		predict_test_label.push_back( PocketPLA( test_vector[ i ] ) );
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
