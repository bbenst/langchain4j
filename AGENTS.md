# Repository Guidelines

## 项目结构与模块组织
该仓库是多模块 Maven 项目，根目录 `pom.xml` 聚合所有子模块。各功能模块以 `langchain4j-*` 命名，源码通常位于 `*/src/main/java`，测试位于 `*/src/test/java`。集成测试集中在 `integration-tests/`，文档位于 `docs/`。

## 构建、测试与本地开发命令
- `make build`：执行 `mvn -U -T12C clean package`，用于全量构建。
- `make lint`：运行 `spotless:check`，检查格式与风格。
- `make format`：运行 `spotless:apply`，自动格式化代码。
- `mvn clean test`：运行所有模块单元测试。
- `./mvnw spotless:check` / `./mvnw spotless:apply`：按贡献指南进行格式检查与修复。

## 编码风格与命名规范
- 兼容 Java 17。
- 遵循既有代码风格与命名约定，避免无必要的重排或大范围格式化。
- 关键格式化工具为 Spotless（通过 Makefile 或 Maven 插件调用）。

## 测试指南
- 变更必须包含单元测试和/或集成测试，覆盖正反用例。
- 部分集成测试需要环境变量提供 API Key（可通过 `EnabledIfEnvironmentVariable` 注解定位）。
- 新模型或存储集成请参考 `langchain4j-core` 与 `langchain4j` 中的 `Abstract*IT` 基类。

## 提交与 PR 指南
- 近期提交信息常见模式：`feat:`、`feature(...)`、`docs:`、`[BUG]`，并在结尾包含 `(#1234)`。
- PR 通常先以 Draft 提交，完整填写模板。
- 保持变更小而聚焦，避免将重构与功能修改混在同一 PR。
- 如有新增集成或功能，补充文档与示例，并运行 `make lint`、`make format`。

## Agent 特别说明
- Java 代码需使用中文注释，类/方法/成员变量必须用 JavaDoc（包含 `@param`、`@return`、`@throws`）。
- 复杂逻辑需在代码上一行添加中文原因说明，简单 getter/setter 可简化方法内注释。
