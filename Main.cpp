extern void jacubi();
extern void jacubi_redblack();
extern void sor();
extern void sor_separated();


int main() {

	switch (3)
	{
	case 0:
		jacubi();
		break;
	case 1:
		jacubi_redblack();
		break;
	case 2:
		sor();
		break;
	case 3:
		sor_separated();
		break;
	default:
		break;
	}

	return 0;
}
