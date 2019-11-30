#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <map> 

using namespace std;

int train_text_num, train_word_num;  //训练数据的文本数量和单词总数 
map< int, string > train_label;  //训练数据文本对应的标签
map< string, int > train_vocabulary;  //记录训练数据中所有单词出现的先后顺序 
vector< double > idf;  //idf
vector< vector< double > > train_tf_idf;  //训练数据文本的tf-idf矩阵
vector< vector< string > > train_data;  //记录训练数据中每篇文章出现了什么单词
 
void SplitString()  
{
	int end_pos = 0;
	string line = "", word = "";
	vector< string > temp;  
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/classification_dataset/train_set.csv" ); 
	
	getline( in, line );
	while( getline( in, line ) )  //按行读取文件 
	{
		end_pos = 0;
		word = "";
		temp.clear();
		for( int i = 0; i < line.size(); i ++ ) 
			if( line[ i ] == ',' )  
		    	end_pos = i;
        for( int i = 0; i <= end_pos; i ++ ) 
        {
        	if( line[ i ] != ' ' && i != end_pos ) 
        	    word += line[ i ];
        	else
        	{
        		if( word != "" )
        		{
        			if( train_vocabulary[ word ] == 0 )   
        			    train_vocabulary[ word ] = ++ train_word_num;  
        		    temp.push_back( word );
        		}
        		word = "";
        	}
        }
        for( int i = end_pos + 1; i < line.size(); i ++ )
            word += line[ i ];
		train_data.push_back( temp );  
		train_label[ train_text_num ++ ] = word;
	}
	  
	in.close();
}

void GetTFIDFMatrix()  
{
	int cnt = 0;
	vector< double > temp; 
    
    for( int i = 0; i < train_word_num; i ++ )  
    {
    	cnt = 0;
    	for( int j = 0; j < train_text_num; j ++ ) 
        {
        	for( int k = 0; k < train_data[ j ].size(); k ++ )   
	        {
	        	if( train_vocabulary[ train_data[ j ][ k ] ] - 1 == i )
	        	{
	        		cnt ++;   
	        		break;
	        	}
	        }
        }
	    idf.push_back( log( 1.0 * train_text_num / ( 1 + cnt ) ) / log( 2 ) );   
    }
	for( int i = 0; i < train_text_num; i ++ ) 
	{
		temp.clear();
		for( int j = 0; j < train_word_num; j ++ )  
	    {
	    	cnt = 0;
	    	for( int k = 0; k < train_data[ i ].size(); k ++ )  
	        	if( train_vocabulary[ train_data[ i ][ k ] ] - 1 == j )
	        		cnt ++;
	        temp.push_back( 1.0 * cnt * idf[ j ] / train_data[ i ].size() );   
	    }
	    train_tf_idf.push_back( temp );
	}
}

double GetDistance( vector< double > train_data_vec, vector< double > test_data_vec )
{
    double train_length = 0, test_length = 0, temp = 0;
	
	if( train_data_vec.size() < test_data_vec.size() )
		for( int i = train_data_vec.size(); i < test_data_vec.size(); i ++ )
		    train_data_vec.push_back( 0 );
	for( int i = 0; i < test_data_vec.size(); i ++ )
	{
		train_length += pow( 1.0 * train_data_vec[ i ], 2 );
		test_length += pow( 1.0 * test_data_vec[ i ], 2 );
		temp += 1.0 * train_data_vec[ i ] * test_data_vec[ i ];
	}
	
	return 0 - temp / ( sqrt( train_length ) * sqrt( test_length ) );
}

string KNNModel( vector< string > test_data )
{
	int k = 11, cnt = 0, max = 0;
	int temp_word_num = train_word_num;
	bool flag = false;
	map< string, int > temp_voc;
	multimap< double, int > vec_distance;
	vector< double > test_vec;
	vector< double > temp_idf;
	int label_cnt[ 6 ];
	
	temp_voc = train_vocabulary;
	temp_idf = idf;
	vec_distance.clear();
	test_vec.clear();
	for( int i = 0; i < 6; i ++ )
	    label_cnt[ i ] = 0; 
	for( int i = 0; i < test_data.size(); i ++ )
	{
		if( temp_voc[ test_data[ i ] ] == 0 )   
        {
        	temp_voc[ test_data[ i ] ] = ++ temp_word_num; 
            temp_idf.push_back( log( 1.0 * ( train_text_num + 1 ) / 2 ) / log( 2 ) ); 
        }
	}
	for( int i = 0; i < temp_word_num; i ++ )  
	{
	    for( int j = 0; j <  test_data.size(); j ++ ) 
	        if( temp_voc[ test_data[ j ] ] - 1 == i )
	        	cnt ++;
	    test_vec.push_back( 1.0 * cnt * temp_idf[ i ] / test_data.size() );
	    cnt = 0;
	}
	for( int i = 0; i < train_text_num; i ++ ) 
		vec_distance.insert( pair< double, int >( GetDistance( train_tf_idf[ i ], test_vec ), i ) );
	for( multimap< double, int > :: iterator it = vec_distance.begin(); cnt < k; cnt ++, it ++ )
	{
		if( train_label[ ( * it ).second ] == "anger" )
		    label_cnt[ 0 ] ++;
		if( train_label[ ( * it ).second ] == "disgust" )
		    label_cnt[ 1 ] ++;
		if( train_label[ ( * it ).second ] == "fear" )
		    label_cnt[ 2 ] ++;
		if( train_label[ ( * it ).second ] == "joy" )
		    label_cnt[ 3 ] ++;
		if( train_label[ ( * it ).second ] == "sad" )
		    label_cnt[ 4 ] ++;
		if( train_label[ ( * it ).second ] == "surprise" )
		    label_cnt[ 5 ] ++;
	}
	for( int i = 0; i < 6; i ++ )
	    if( label_cnt[ max ] <= label_cnt[ i ] )
	        max = i;
	if( max == 0 )
	    return "anger";
	if( max == 1 )
	    return "disgust";
	if( max == 2 )
	    return "fear";
	if( max == 3 )
	    return "joy";
	if( max == 4 )
	    return "sad";
	if( max == 5 )
	    return "surprise";
	return "error";
}

void OptimizeModel()
{
	int end_pos = 0, cnt = 0, sum = 0;
	string line = "", word = "";
	vector< string > temp;  
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/classification_dataset/validation_set.csv" ); 
	
	getline( in, line );
	while( getline( in, line ) )  //按行读取文件 
	{
		sum ++;
		end_pos = 0;
		word = "";
		temp.clear();
		for( int i = 0; i < line.size(); i ++ ) 
			if( line[ i ] == ',' )  
		    	end_pos = i;
        for( int i = 0; i <= end_pos; i ++ ) 
        {
        	if( line[ i ] != ' ' && i != end_pos ) 
        	    word += line[ i ];
        	else
        	{
        		if( word != "" )
        		    temp.push_back( word );
        		word = "";
        	}
        }
        for( int i = end_pos + 1; i < line.size(); i ++ )
            word += line[ i ];
        if( KNNModel( temp ) == word )
            cnt ++;
	}
	cout << "验证集文本总数：" << sum << " 预测正确的标签数量：" << cnt << endl;
	cout << "准确率： " << 1.0 * cnt / sum << endl;
	  
	in.close();
}

void PredictLabel() 
{
	bool flag = false;
	int start_pos = 0, end_pos = 0, cnt = 0, textid = 0;
	string line = "", word = "";
	vector< string > temp;  
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/classification_dataset/test_set.csv" ); 
	ofstream out( "15352076_diaohongjin_KNN_classification.csv" );
	
	getline( in, line );
	out << "textid,label" << endl;
	while( getline( in, line ) )  //按行读取文件 
	{
		textid ++;
		flag = false;
		start_pos = 0;
		end_pos = 0;
		word = "";
		temp.clear();
		for( int i = 0; i < line.size(); i ++ ) 
		{
			if( line[ i ] == ',' && ! flag ) 
			{
				start_pos = i + 1;
				flag = true;
			} 
			else if( line[ i ] == ',' && flag )
		    	end_pos = i;
		}
        for( int i = start_pos; i <= end_pos; i ++ ) 
        {
        	if( line[ i ] != ' ' && i != end_pos ) 
        	    word += line[ i ];
        	else
        	{
        		if( word != "" )
        		    temp.push_back( word );
        		word = "";
        	}
        }
        out << textid << "," << KNNModel( temp ) << endl;
	}
	  
	in.close();
	out.close();
}

int main()
{
	int choice; 
	
	SplitString();
	GetTFIDFMatrix();
	
	cout << "读取验证集请输入1，读取测试集请按2" << endl << "您的选择是：";
	cin >> choice;	
	if( choice == 1 )
	    OptimizeModel();
	if( choice == 2 )
		PredictLabel();
	
	return 0;
}
