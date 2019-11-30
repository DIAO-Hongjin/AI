#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <map> 
#include <set>

using namespace std;

int train_text_num, train_word_num;  //训练数据的文本数量和单词总数
map< string, double > train_label_prob;
map< string, int > train_label_sum;
map< string, set< string > > train_label_word;
map< pair< string, string >, int > train_word_frq;
map< int, string > train_label;  //训练数据文本对应的标签
map< string, int > train_vocabulary;  //记录训练数据中所有单词出现的先后顺序 
vector< vector< double > > train_tf; 
vector< vector< string > > train_data;  //记录训练数据中每篇文章出现了什么单词
 
void SplitString()  
{
	int end_pos = 0;
	string line = "", word = "", label = "";
	vector< string > temp_vec;  
	map< string, int > train_label_num;
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/classification_dataset/train_set.csv" ); 
	
	train_label_num.clear();
	getline( in, line );
	while( getline( in, line ) )  //按行读取文件 
	{
		end_pos = 0;
		word = "";
		label = "";
		temp_vec.clear();
		for( int i = 0; i < line.size(); i ++ ) 
			if( line[ i ] == ',' )  
		    	end_pos = i;
		for( int i = end_pos + 1; i < line.size(); i ++ )
            label += line[ i ];
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
        		    temp_vec.push_back( word );
        		    train_label_word[ label ].insert( word );
        		}
        		word = "";
        	}
        }
		train_data.push_back( temp_vec );  
		train_label[ train_text_num ++ ] = label;
		train_label_num[ label ] ++;
		train_label_sum[ label ] += temp_vec.size();
	}
	for( map< string, int > :: iterator it = train_label_num.begin(); it != train_label_num.end(); it ++ )
	    train_label_prob[ it -> first ] = ( it -> second ) * 1.0 / train_text_num;
	  
	in.close();
}

void GetTF()  
{
	for( int i = 0; i < train_text_num; i ++ )  //遍历训练集文本 
		for( int j = 0; j < train_data[ i ].size(); j ++ )  //遍历训练集文本中的每一个单词  
			train_word_frq[ pair< string, string >( train_data[ i ][ j ], train_label[ i ] ) ] ++;  //计录这个单词在这个标签下出现的次数
}

string NBModel( vector< string > test_data )
{
	double a = 0.55;
	double max = -10000000;
	string result = "";
	map< string, double > test_prob;
	
	test_prob.clear();
	for( int i = 0; i < test_data.size(); i ++ )
	{
	    for( map< string, int > :: iterator it = train_label_sum.begin(); it != train_label_sum.end(); it ++ )
	    {
	    	if( train_vocabulary[ test_data[ i ] ] > 0 )
	    		test_prob[ it -> first ] += log( ( train_word_frq[ pair< string, string >( test_data[ i ], it -> first ) ] + a ) * 1.0 / ( it -> second + a * train_word_num ) );
	    	else
	    	    test_prob[ it -> first ] += 0;
	    }
	}
	for( map< string, double > :: iterator it = test_prob.begin(); it != test_prob.end(); it ++ )
	    it -> second += log( train_label_prob[ it -> first ] );
	for( map< string, double > :: iterator it = test_prob.begin(); it != test_prob.end(); it ++ )
	{
		if( max < ( it -> second ) )
	    {
	    	max = ( it -> second );
	        result = it -> first;
	    }
	}
	
	return result;
}

void OptimizeModel()
{
	int end_pos = 0, cnt = 0, sum = 0;
	string line = "", word = "", label = "";
	vector< string > temp;  
	ifstream in( "F://AI/lab2(KNN+NB)/DATA/classification_dataset/validation_set.csv" ); 

	getline( in, line );
	while( getline( in, line ) )  //按行读取文件 
	{
		sum ++;
		end_pos = 0;
		word = "";
		label = "";
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
            label += line[ i ];
        if( NBModel( temp ) == label )
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
	ofstream out( "15352076_diaohongjin_NB_classification.csv" );

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
        out << textid << "," << NBModel( temp ) << endl;
	}
	  
	in.close();
	out.close();
}

int main()
{
	int choice;
	 
	SplitString();
	GetTF();
	
	cout << "读取验证集请输入1，读取测试集请按2" << endl << "您的选择是：";
	cin >> choice;	
	if( choice == 1 )
	    OptimizeModel();
	if( choice == 2 )
		PredictLabel();

	return 0;
} 
