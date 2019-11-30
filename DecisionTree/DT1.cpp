#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream> 
#include <cmath>
#include <algorithm>

using namespace std;

template< class out_type, class in_type >  
out_type convert( const in_type & value )   //任意类型转换 
{
    stringstream stream;  
    out_type result;
    
    stream << value;  
    stream >> result; 
    
    return result;
}

struct Tree  //决策树 
{
	bool const_cum;
	int root_attr;  //根节点属性
	int leaf_label;  //叶节点标签 
	vector< double > leaf_weight;  //叶节点权重 
	vector< int > attr_value;  //属性的取值 
	vector< struct Tree > children;  //子结点 
	
	Tree()  //默认构造函数 
	{
		const_cum = false;
		root_attr = -1;
		leaf_label = 0;
		leaf_weight.clear();
		attr_value.clear();
		children.clear();
	}
};

struct Cnt  //属性计数器 
{
	int all_cnt;  //计算该属性出现的总次数 
	map< int, int > label_cnt;  //计算该属性在某标签出现的次数 
	
	void clear()  //重置 
	{
		all_cnt = 0;
		label_cnt.clear();
	}
};

vector< vector< int > > all_data;  //输入数据集的所有数据 
vector< int > all_label;  //输入数据集的所有数据对应的标签 
set< int > label_type;  //输入数据集所有标签的种类
map< int, set< int > > attr_type;  //输入数据的所有属性的所有取值 
vector< int > data_id;  //输入数据集的所有数据对应的id 
vector< int > attr_id;  //输入数据集的所有属性
vector< int > train_id;  //训练集数据对应的id 
vector< int > validation_id;  //验证集数据对应的id 

void ReadALLData()  //读取输入数据 
{
	bool flag = true;
	int temp_data_cnt = 0, temp_attr_cnt = 0;
	string line = "", num = "";
	vector< int > temp_vec;
	ifstream in( "F://AI/lab4_Decision_Tree/train.csv" ); 
	
	while( getline( in, line ) ) 
	{
		temp_vec.clear();
		for( int i = 0; i <= line.size(); i ++ )
		{
			if( line[ i ] == ',' )
			{
				//temp_vec.push_back( convert< int, string >( num ) );
				if( convert< int, string >( num ) > 15 )
				{
					if( convert< int, string >( num ) <= 20 )
						temp_vec.push_back( 20 );
					else if( convert< int, string >( num ) <= 25 )
						temp_vec.push_back( 25 );
					else if( convert< int, string >( num ) <= 30 )
						temp_vec.push_back( 30 );
					else if( convert< int, string >( num ) <= 35 )
						temp_vec.push_back( 35 );
					else if( convert< int, string >( num ) <= 40 )
						temp_vec.push_back( 40 );
					else if( convert< int, string >( num ) <= 45 )
						temp_vec.push_back( 45 );
					else 
						temp_vec.push_back( 50 );
				}
				else
					temp_vec.push_back( convert< int, string >( num ) );
				num = "";
				if( flag )
					attr_id.push_back( temp_attr_cnt ++ );
			}
			else if( i == line.size() )
			{
				all_label.push_back( convert< int, string >( num ) );
				label_type.insert( convert< int, string >( num ) );
				num = "";
			}
			else
				num += line[ i ];
		}
		all_data.push_back( temp_vec ); 
		data_id.push_back( temp_data_cnt ++ );
		flag = false;
	}
	for( int i = 0; i < attr_id.size(); i ++ )
	    for( int j = 0; j < data_id.size(); j ++ )
	        attr_type[ i ].insert( all_data[ j ][ i ] );
	
	in.close();
}  

void PartitionDataSet()
{
	for( int i = 0; i < data_id.size(); i ++ )
	{
		if( i % 10 == 0 )
		    validation_id.push_back( i );
		else
		    train_id.push_back( i );
	}
}

map< int, map< int, struct Cnt > > GetAttrInfo( vector< int > cur_data_id, vector< int > cur_attr_id )
{
	int temp_all_cnt = 0;
	struct Cnt temp_cnt;
	vector< int > temp_vec;
	map< int, int > temp_label_cnt;
	map< int, struct Cnt > temp_attr;
	map< int, map< int, struct Cnt > > attr_info;
	
	attr_info.clear();
	for( int i = 0; i < cur_attr_id.size(); i ++ )
	{
		temp_attr.clear();
		for( set< int > :: iterator it = attr_type[ cur_attr_id[ i ] ].begin(); it != attr_type[ cur_attr_id[ i ] ].end(); it ++ )
		{
			temp_all_cnt = 0;
			temp_label_cnt.clear();
			temp_cnt.clear();
			for( int j = 0; j < cur_data_id.size(); j ++ )
			{
				if( all_data[ cur_data_id[ j ] ][ cur_attr_id[ i ] ] == *it )
			    {
			    	temp_all_cnt ++;
			    	for( set< int > :: iterator temp_it = label_type.begin(); temp_it != label_type.end(); temp_it ++ )
			    	    if( ( *temp_it ) == all_label[ cur_data_id[ j ] ] )
			    	        temp_label_cnt[ *temp_it ] ++;
			    }
			}
			temp_cnt.all_cnt = temp_all_cnt; 
			temp_cnt.label_cnt = temp_label_cnt;
			temp_attr[ *it ] = temp_cnt;
		}
		attr_info[ cur_attr_id[ i ] ] = temp_attr;
	}
	
	return attr_info;
}

double GetHD( vector< int > cur_data_id )
{
	double hd = 0;
	map< int, int > label_sum;
	
	label_sum.clear();
	for( set< int > :: iterator it = label_type.begin(); it != label_type.end(); it ++ )
	{
		for( int i = 0; i < cur_data_id.size(); i ++ )
	    {
	    	if( all_label[ cur_data_id[ i ] ] == ( *it ) )
	            label_sum[ *it ] ++;
	        else
	            label_sum[ *it ] += 0;
	    }
	}
	for( map< int, int > :: iterator it = label_sum.begin(); it != label_sum.end(); it ++ )
	    hd -= ( ( it -> second ) * 1.0 / cur_data_id.size() ) * ( log( ( ( it -> second ) * 1.0 / cur_data_id.size() ) ) / log( 2 ) );
	
	return hd;
}

double GetHDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	double hda = 0, temp = 0;
	
	for( map< int, struct Cnt > :: iterator it1 = attr_info[ current_attr ].begin(); it1 != attr_info[ current_attr ].end(); it1 ++ )
	{
		temp = 0;
		for( map< int, int > :: iterator it2 = ( it1 -> second ).label_cnt.begin(); it2 != ( it1 -> second ).label_cnt.end(); it2 ++ )
			temp -= ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ) * ( log( ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ) ) / log( 2 ) );
		temp *= ( ( it1 -> second ).all_cnt ) * 1.0 / cur_data_id.size();
		hda += temp;
	}
	
	return hda;
}

double GetGDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	double gda = GetHD( cur_data_id ) - GetHDA( cur_data_id, attr_info, current_attr );
	
	return gda;
}

double GetSplitInfoDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )
{
	double sida = 0;
	
	for( map< int, struct Cnt > :: iterator it = attr_info[ current_attr ].begin(); it != attr_info[ current_attr ].end(); it ++ )
	    if( ( it -> second ).all_cnt == 0 )
	        sida -= 0;
	    else
		    sida -= ( ( ( it -> second ).all_cnt ) * 1.0 / cur_data_id.size() ) * ( log( ( ( it -> second ).all_cnt ) * 1.0 / cur_data_id.size() ) / log( 2 ) );
	
	return sida;
}

double GetGRatioA( vector< int > cur_data_id, vector< int > cur_attr_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )  
{
	double gra = 0, avg_sida = 0;
	
	for( int i = 0; i < cur_attr_id.size(); i ++ )
	    avg_sida += GetSplitInfoDA( cur_data_id, attr_info, cur_attr_id[ i ] );
	avg_sida /= 1.0 * cur_attr_id.size();
	gra = GetGDA( cur_data_id, attr_info, current_attr ) / ( avg_sida + GetSplitInfoDA( cur_data_id, attr_info, current_attr ) );
	
	return gra;
}

double GetGiniDA( vector< int > cur_data_id, map< int, map< int, struct Cnt > > attr_info, int current_attr )   
{
	double ginida = 0, temp = 0;
	
	for( map< int, struct Cnt > :: iterator it1 = attr_info[ current_attr ].begin(); it1 != attr_info[ current_attr ].end(); it1 ++ )
	{
		temp = 1;
		for( map< int, int > :: iterator it2 = ( it1 -> second ).label_cnt.begin(); it2 != ( it1 -> second ).label_cnt.end(); it2 ++ )
			temp -= pow( ( it2 -> second ) * 1.0 / ( ( it1 -> second ).all_cnt ), 2 );
		temp *= ( ( it1 -> second ).all_cnt ) * 1.0 / cur_data_id.size();
		ginida += temp;
	}
	
	return ginida;
}

set< int > FindLabelType( vector< int > cur_data_id )
{
	set< int > result_set;
	
	result_set.clear();
	for( int i = 0; i < cur_data_id.size(); i ++ )
		result_set.insert( all_label[ cur_data_id[ i ] ] );
	
	return result_set;
}


double GetDotProduct( vector< double > vec1, vector< int > vec2, int length )
{
	double result = 0;;
	
	for( int i = 0; i < length; i ++ )
		result += 1.0 * vec1[ i ] * vec2[ i ];
	
	return result; 
}

int SignFunction( double num )
{
	if( num > 0 )
	    return 1;
	if( num <= 0 )
	    return -1;
}

vector< double > RenewWeight( vector< double > old_weight, vector< int > x, int y, int length )
{
	vector< double > result;
	
	result.clear();
	for( int i = 0; i < length; i ++ )
	    result.push_back( old_weight[ i ] + y * 1.0 * x[ i ] );
	
	return result;
}

vector< double > GetWeight( vector< int > cur_data_id )  //计算叶子结点的PLA权重 
{
	int cnt = 0, iterator = 10000; 
	vector< double > weight;
	vector< int > temp;
	
	weight.clear();
	//初始化权重向量 
	for( int i = 0; i < attr_id.size() + 1; i ++ )  
	    weight.push_back( 0 );
	//遍历划分到该结点的样本，训练模型 
	for( int i = 0; i < cur_data_id.size(); i ++ )  
	{
		temp.clear();
	    //样本前面加上1 
		temp.push_back( 1 );
		for( int j = 0; j < attr_id.size(); j ++ )
		    temp.push_back( all_data[ cur_data_id[ i ] ][ attr_id[ j ] ] );
		//如果训练数据的预测标签与实际标签不同，更新权重，迭代一定次数 
		if( SignFunction( GetDotProduct( weight, temp, attr_id.size() + 1 ) ) != all_label[ cur_data_id[ i ] ] )
		{
			weight = RenewWeight( weight, temp, all_label[ cur_data_id[ i ] ], attr_id.size() + 1 );
			if( cnt < iterator )   
			{
				i = -1;
				cnt ++;
			}
		}
	}
	
	return weight;  //返回训练好的权重 
}

int ChooseRootAttr( string way, vector< int > cur_data_id, vector< int > cur_attr_id )
{
	int root_attr = 0;
	double max = -10000, min = 10000, temp = 0;
	map< int, map< int, struct Cnt > > attr_info = GetAttrInfo( cur_data_id, cur_attr_id );
	
	if( way == "ID3" )
	{
		for( int i = 0; i < cur_attr_id.size(); i ++ )
		{
			temp = GetGDA( cur_data_id, attr_info, cur_attr_id[ i ] );
			if( max < temp )
		    {
	    		max = temp;
	    		root_attr = cur_attr_id[ i ];
		    }
		}
	}
	else if( way == "C4.5" )
	{
		for( int i = 0; i < cur_attr_id.size(); i ++ )
		{
			temp = GetGRatioA( cur_data_id, cur_attr_id, attr_info, cur_attr_id[ i ] );
			if( max < temp )
		    {
	    		max = temp;
	    		root_attr = cur_attr_id[ i ];
		    }
		}
	}
	else if( way == "CART" )
	{
		for( int i = 0; i < cur_attr_id.size(); i ++ )
		{
			temp = GetGiniDA( cur_data_id, attr_info, cur_attr_id[ i ] );
			if( min > temp )
		    {
	    		min = temp;
	    		root_attr = cur_attr_id[ i ];
		    }
		}
	}
	
	return root_attr;
}

void CreateTree( string way, vector< int > far_data_id, vector< int > cur_data_id, vector< int > cur_attr_id, struct Tree &tree )
{
	vector< int > new_data_id, new_attr_id;
	set< int > res_label_type = FindLabelType( cur_data_id );
	
	//当该结点的数据集只剩下一个标签时，将结点设置为叶子结点，标签为剩下的这个标签
	if( res_label_type.size() == 1 )
	{
		for( set< int > :: iterator it = res_label_type.begin(); it != res_label_type.end(); it ++ )
		    tree.leaf_label = *it;
		tree.const_cum = true;  //标记该叶结点的标签为确定的标签 
		return ;
	}
	//当该结点上的数据集是空集时，将结点设置为叶子结点，叶结点的模型为PLA求得的权重 
	if( cur_data_id.size() == 0 )
	{
		//以父结点数据集为训练集得到的PLA权重模型 
		tree.leaf_weight = GetWeight( far_data_id );  
		return ;
	}
	//当属性集为空时，将结点设置为叶子结点，叶结点的模型为PLA求得的权重
	if( cur_attr_id.size() == 0 )
	{
		//以当前结点数据集为训练集得到的PLA权重模型
		tree.leaf_weight = GetWeight( cur_data_id );
		return ;
	}
	//训练数据太少得到的PLA权重没有什么意义，进行简单剪枝
	//当数据集样本数量小于20个时决策树停止生长 
	if( cur_data_id.size() <= 20 )
	{
		//以当前结点数据集为训练集得到的PLA权重模型
		tree.leaf_weight = GetWeight( cur_data_id );
		return ;
	} 
	tree.root_attr = ChooseRootAttr( way, cur_data_id, cur_attr_id );
	for( set< int > :: iterator it = attr_type[ tree.root_attr ].begin(); it != attr_type[ tree.root_attr ].end(); it ++ )
	{
		new_data_id.clear();
		new_attr_id.clear();
		for( int j = 0; j < cur_attr_id.size(); j ++ )
		    if( cur_attr_id[ j ] != tree.root_attr )
		        new_attr_id.push_back( cur_attr_id[ j ] );
		for( int j = 0; j < cur_data_id.size(); j ++ )
		    if( all_data[ cur_data_id[ j ] ][ tree.root_attr ] == ( *it ) )
		        new_data_id.push_back( cur_data_id[ j ] ); 
		struct Tree new_tree;
		CreateTree( way, cur_data_id, new_data_id, new_attr_id, new_tree );
		tree.attr_value.push_back( *it );
		tree.children.push_back( new_tree );
	} 
}

int PredictLabel( vector< int > data, struct Tree tree )  //利用生成的决策树预测测试数据data的标签 
{
	if( tree.root_attr == -1 && tree.const_cum )  //确定标签的叶子结点：返回叶子结点的标签
		return tree.leaf_label;
	
	if( tree.root_attr == -1 && ! tree.const_cum )  //不确定标签的叶子结点：用PLA权重预测标签 
		return SignFunction( GetDotProduct( tree.leaf_weight, data, attr_id.size() + 1 ) ); 
	
	for( int i = 0; i < tree.attr_value.size(); i ++ )  //非叶子结点：递归找到对应的叶子结点
	    if( data[ tree.root_attr ] == tree.attr_value[ i ] )
	        return PredictLabel( data, tree.children[ i ] );
	
	return *( label_type.begin() );//到这里说明测试数据的某个属性出现了训练集中不存在的取值，返回指定标签 
}

void Validation( string way, vector< int > validation_id, struct Tree tree )
{
	int cnt = 0;
	
	for( int i = 0; i < validation_id.size(); i ++ )
	{
		if( all_label[ validation_id[ i ] ] == PredictLabel( all_data[ validation_id[ i ] ], tree ) )
			cnt ++;
		//cout << "预测标签：" << PredictLabel( all_data[ validation_id[ i ] ], tree ) << " 正确标签：" << all_label[ validation_id[ i ] ] << endl;
	}
	
	cout << way << ": 正确率为" << cnt * 1.0 / validation_id.size() << endl;
}

int main()
{
    struct Tree ID3_tree, C45_tree, CART_tree;
    
	ReadALLData();
	PartitionDataSet();
	
	CreateTree( "ID3", train_id, train_id, attr_id, ID3_tree );
	CreateTree( "C4.5", train_id, train_id, attr_id, C45_tree );
	CreateTree( "CART", train_id, train_id, attr_id, CART_tree );
	
	Validation( "ID3", validation_id, ID3_tree );
	Validation( "C4.5", validation_id, C45_tree );
	Validation( "CART", validation_id, CART_tree );
	
	return 0;
}


