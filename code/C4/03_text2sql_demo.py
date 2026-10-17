"""
Text2SQL 框架演示脚本。

该脚本会：
1. 检查 DeepSeek API Key；
2. 创建一个 SQLite 演示数据库；
3. 初始化 Text2SQL 代理并加载知识库；
4. 使用自然语言问题查询数据库；
5. 最后清理资源并删除演示数据库。
"""

import os
import sys
import sqlite3

from dotenv import load_dotenv
# 从 .env 文件加载环境变量（例如 DEEPSEEK_API_KEY）
load_dotenv()

# 将当前脚本所在目录下的 text2sql 子目录加入模块搜索路径，
# 便于后续导入 text2sql 相关模块。
sys.path.append(os.path.join(os.path.dirname(__file__), 'text2sql'))

from text2sql.text2sql_agent import SimpleText2SQLAgent


def setup_demo():
    """设置演示环境。

    返回:
        tuple: 成功时返回 (agent, db_path)，即 Text2SQL 代理和数据库路径。
        None: 当 API Key 缺失、数据库连接失败或知识库加载失败时返回 None。
    """
    print("=== Text2SQL框架演示 ===\n")

    # 从环境变量中读取 DeepSeek API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("先设置DEEPSEEK_API_KEY环境变量")
        return None

    # 创建演示用的 SQLite 数据库，并返回数据库文件路径
    print("创建演示数据库...")
    db_path = create_demo_database()

    # 初始化 Text2SQL 代理，后续自然语言查询都通过该代理完成
    print("初始化Text2SQL代理...")
    agent = SimpleText2SQLAgent(api_key=api_key)

    # 让代理连接刚刚创建的 SQLite 数据库
    print("连接数据库...")
    if not agent.connect_database(db_path):
        print("数据库连接失败!")
        return None

    # 加载知识库，通常包含表结构、字段说明、业务术语等上下文信息
    print("加载知识库...")
    try:
        agent.load_knowledge_base()
        print("知识库加载成功!")
    except Exception as e:
        print(f"知识库加载失败: {str(e)}")
        return None

    return agent, db_path


def create_demo_database():
    """创建演示数据库。

    如果数据库文件已存在，则先删除，再重新创建表并插入示例数据。
    """
    db_path = "text2sql_demo.db"

    # 如果旧的演示数据库存在，先删除，保证每次运行都是干净环境
    if os.path.exists(db_path):
        os.remove(db_path)

    # 连接 SQLite 数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建用户表
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER,
            city TEXT
        )
    """)

    # 创建产品表
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER
        )
    """)

    # 创建订单表
    # user_id 和 product_id 分别关联 users 表和 products 表
    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            order_date TEXT,
            total_price REAL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    # 准备用户示例数据
    users_data = [
        (1, '张三', 'zhangsan@email.com', 25, '北京'),
        (2, '李四', 'lisi@email.com', 32, '上海'),
        (3, '王五', 'wangwu@email.com', 28, '广州'),
        (4, '赵六', 'zhaoliu@email.com', 35, '深圳'),
        (5, '陈七', 'chenqi@email.com', 29, '杭州'),
    ]

    # 准备产品示例数据
    products_data = [
        (1, 'iPhone 15', '电子产品', 7999.0, 50),
        (2, 'MacBook Pro', '电子产品', 12999.0, 20),
        (3, 'Nike运动鞋', '服装', 599.0, 100),
        (4, '办公椅', '家具', 899.0, 30),
        (5, '台灯', '家具', 199.0, 80),
        (6, 'iPad', '电子产品', 3999.0, 40),
        (7, 'Adidas外套', '服装', 399.0, 60),
    ]

    # 准备订单示例数据
    orders_data = [
        (1, 1, 1, 1, '2024-01-15', 7999.0),
        (2, 2, 3, 2, '2024-01-16', 1198.0),
        (3, 3, 5, 1, '2024-01-17', 199.0),
        (4, 1, 2, 1, '2024-01-18', 12999.0),
        (5, 4, 4, 1, '2024-01-19', 899.0),
        (6, 5, 6, 1, '2024-01-20', 3999.0),
        (7, 2, 7, 1, '2024-01-21', 399.0),
    ]

    # 批量插入示例数据
    cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", users_data)
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products_data)
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders_data)

    # 提交事务并关闭数据库连接
    conn.commit()
    conn.close()

    print(f"演示数据库已创建: {db_path}")
    return db_path


def run_demo_queries(agent):
    """运行演示查询。

    参数:
        agent: 已经连接数据库并加载知识库的 Text2SQL 代理。
    """
    # 用于演示的自然语言问题列表
    demo_questions = [
        "查询所有用户的姓名和邮箱",
        "年龄大于30的用户有哪些",
        "哪些产品的库存少于50",
        "查询来自北京的用户的所有订单",
        "统计每个城市的用户数量",
        "查询价格在500-8000之间的产品"
    ]

    print("\n开始运行演示查询...\n")

    # 统计成功执行的查询数量
    success_count = 0

    # 遍历每一个演示问题，编号从 1 开始
    for i, question in enumerate(demo_questions, 1):
        print(f"问题 {i}: {question}")
        print("-" * 60)

        try:
            # 调用 Text2SQL 代理执行自然语言查询
            result = agent.query(question)

            # result 通常包含 success、sql、results、error 等字段
            if result["success"]:
                print(f"成功! SQL: {result['sql']}")

                # 如果结果中包含 rows，则说明是表格型查询结果
                if isinstance(result["results"], dict) and "rows" in result["results"]:
                    count = result["results"]["count"]
                    print(f"返回 {count} 行数据")

                    # 只显示前 2 行数据，避免输出过长
                    if count > 0:
                        for j, row in enumerate(result["results"]["rows"][:2]):
                            row_str = " | ".join(f"{k}: {v}" for k, v in row.items())
                            print(f"  {j + 1}. {row_str}")

                        # 如果数据超过 2 行，提示还有多少行未显示
                        if count > 2:
                            print(f"  ... 还有 {count - 2} 行")
                else:
                    # 非表格型结果直接打印
                    print(f"结果: {result['results']}")

                success_count += 1

            else:
                # 查询失败时打印错误信息和生成的 SQL
                print(f"失败: {result['error']}")
                print(f"SQL: {result['sql']}")

        except Exception as e:
            # 捕获执行过程中的异常，避免单个问题失败导致整个演示中断
            print(f"执行错误: {str(e)}")

        print()

    # 演示问题总数
    # 注意：原代码此处未继续输出 success_count / total_count 的统计结果
    total_count = len(demo_questions)


def cleanup(agent, db_path):
    """清理资源。

    参数:
        agent: Text2SQL 代理实例，可能为 None。
        db_path: 演示数据库路径。
    """
    print("\n清理资源...")

    # 如果代理存在，则释放代理相关资源
    if agent:
        agent.cleanup()

    # 删除演示数据库文件
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"已删除演示数据库: {db_path}")


def main():
    """主函数：组织演示流程。"""
    # 设置演示环境
    setup_result = setup_demo()

    # 如果环境设置失败，直接结束
    if setup_result is None:
        return

    agent, db_path = setup_result

    try:
        # 运行演示查询
        run_demo_queries(agent)

    finally:
        # 无论演示是否成功，都确保清理资源
        cleanup(agent, db_path)


if __name__ == "__main__":
    # 当脚本被直接运行时，执行主函数
    main()